import argparse
import copy
import json
from datetime import datetime
import os
import sys
import shutil, errno
import genesis as gs
import numpy as np
import multiprocessing
from PIL import Image
import threading, time

current_directory = os.getcwd()
sys.path.append(current_directory)

from agents.meeting_challenge import *
from agents.memory import SemanticMemory
from ViCo.env import VicoEnv, AgentProcess
from ViCo.modules import *

keep_running = False

def gpu_occupy_loop():
    import torch
    if not torch.cuda.is_available():
        return
    while keep_running:
        a = torch.randn(4096, 4096, device='cuda')
        _ = torch.mm(a, a)
        time.sleep(0.5)

def end_processes():
    for p in multiprocessing.active_children():
        print(f"[Multiprocessing] PID={p.pid} Name={p.name}")
        p.terminate()
        p.join()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", type=str, default='32')
    parser.add_argument("--logging_level", type=str, default='info')
    parser.add_argument("--backend", type=str, default='cpu')
    parser.add_argument("--head_less", '-l', action='store_true')
    parser.add_argument("--multi_process", '-m', action='store_true')
    parser.add_argument("--output_dir", "-o", type=str, default='ViCo/meeting_challenge/output')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--overwrite", action='store_true')

    ### Simulation configurations
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--enable_collision", action='store_true')
    parser.add_argument("--skip_avatar_animation", action='store_true')
    parser.add_argument("--enable_gt_segmentation", action='store_true')
    parser.add_argument("--max_seconds", type=int, default=86400) # 24 hours
    parser.add_argument("--save_per_seconds", type=int, default=10)
    parser.add_argument("--enable_third_person_cameras", action='store_true')
    parser.add_argument("--curr_time", type=str)
    parser.add_argument("--start_id", type=int)
    parser.add_argument("--only_one_sample", action='store_true')

    ### Scene configurations
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--enable_indoor_scene", action='store_true')
    parser.add_argument("--enable_indoor_activities", action='store_true')
    parser.add_argument("--enable_outdoor_objects", action='store_true')
    parser.add_argument("--outdoor_objects_assets_dir", type=str, default='ViCo/scene/object_assets')
    parser.add_argument("--outdoor_objects_max_num", type=int, default=10)
    parser.add_argument("--no_load_scene", action='store_true')

    # Traffic configurations
    parser.add_argument("--no_traffic_manager", action='store_true')
    parser.add_argument("--tm_vehicle_num", type=int, default=0)
    parser.add_argument("--tm_avatar_num", type=int, default=0)
    parser.add_argument("--enable_tm_debug", action='store_true')

    ### Agent configurations
    parser.add_argument("--config", type=str, default='agents_num_25')
    parser.add_argument("--agent_type", type=str, choices=['heuristic', 'llm', 'mcts', 'random'])
    parser.add_argument("--agent_type2", type=str, choices=['heuristic', 'llm', 'mcts', 'random'])
    parser.add_argument("--no_react", action='store_true')
    parser.add_argument("--lm_source", type=str, choices=["openai", "azure", "huggingface"], default="azure", help="language model source")
    parser.add_argument("--lm_id", "-lm", type=str, default="gpt-35-turbo", help="language model id")
    parser.add_argument("--max_tokens", type=int, default=4096, help="maximum tokens")
    parser.add_argument("--temperature", "-t", type=float, default=0, help="temperature")
    parser.add_argument("--top_p", type=float, default=1, help="top p")

    # assistant challenge
    parser.add_argument("--robot_as_agent", action='store_true')
    parser.add_argument("--enable_demo_camera", action='store_true')
    parser.add_argument("--step_limit", type=int, default=1500)
    parser.add_argument("--robot_policy_path", type=str, default="", help="Where to load robot policy")
    args = parser.parse_args()

    random.seed(time.time())
    # Make output directories
    if args.agent_type == 'llm' and args.lm_id != 'gpt-4o':
        args.output_dir = os.path.join(args.output_dir, args.scene,
                                       f"{args.agent_type}-{args.lm_id.split('/')[0]}")
    else:
        if args.agent_type2:
            agent_type = f"{args.agent_type}-{args.agent_type2}"
        else:
            agent_type = args.agent_type
        args.output_dir = os.path.join(args.output_dir, args.scene, f"{agent_type}")
    os.makedirs(args.output_dir, exist_ok=True)
    output_dir = args.output_dir
    result_path = os.path.join(output_dir, "result.json")
    if os.path.exists(result_path):
        result = json.load(open(result_path, 'r'))
        print(f"results exists: {result}")
        if result["done"]:
            print(f"it's already done. Skip running simulation")
            end_processes()
            exit(0)
    if args.overwrite and os.path.exists(args.output_dir):
        print(f"Overwrite the output directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'debug'), exist_ok=True)

    # Read and initialize scene configuration
    os.makedirs(args.output_dir, exist_ok=True)
    config_path = os.path.join(args.output_dir, 'curr_sim')
    if not os.path.exists(config_path):
        seed_config_path = os.path.join('ViCo/assets/scenes', args.scene, args.config)
        print(f"Initiate new simulation from config: {seed_config_path}")
        try:
            shutil.copytree(seed_config_path, config_path)
        except OSError as exc:  # python >2.5
            if exc.errno in (errno.ENOTDIR, errno.EINVAL):
                shutil.copy(seed_config_path, config_path)
            else:
                raise
    else:
        print(f"Continue simulation from config: {config_path}")
    config = json.load(open(os.path.join(config_path, "config.json"), 'r'))
    num_agents = config["num_agents"]

    if args.debug:
        args.enable_third_person_cameras = True
    env = VicoEnv(
        seed=args.seed,
        precision=args.precision,
        logging_level=args.logging_level,
        backend=gs.cpu if args.backend == 'cpu' else gs.gpu,
        head_less=args.head_less,
        resolution=args.resolution,
        challenge='meeting',
        num_agents=config["num_agents"],
        config_path=config_path,
        scene=args.scene,
        enable_indoor_scene=args.enable_indoor_scene,
        enable_outdoor_objects=args.enable_outdoor_objects,
        outdoor_objects_max_num=args.outdoor_objects_max_num,
        enable_collision=args.enable_collision,
        skip_avatar_animation=args.skip_avatar_animation,
        enable_gt_segmentation=args.enable_gt_segmentation,
        no_load_scene=args.no_load_scene,
        output_dir=output_dir,
        enable_third_person_cameras=args.enable_third_person_cameras,
        no_traffic_manager=args.no_traffic_manager,
        enable_tm_debug=args.enable_tm_debug,
        tm_vehicle_num=args.tm_vehicle_num,
        tm_avatar_num=args.tm_avatar_num,
        save_per_seconds=args.save_per_seconds,
        debug=args.debug,
    )
    obs = env.reset()

    # Initialize the proposer agents and NPC agents
    name2idx = {}
    all_agent_processes: list[AgentProcess] = []
    all_agent_name=[]
    for i in range(num_agents):
        basic_kwargs = dict(
            name=config['agent_names'][i],
            pose=config["agent_poses"][i],
            info=config["agent_infos"][i],
            sim_path=config_path,
            no_react=args.no_react,
            debug=args.debug,
            logging_level=args.logging_level,
            multi_process=args.multi_process,
        )
        llm_kwargs = dict(
            lm_source=args.lm_source,
            lm_id=args.lm_id,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        if agent_type == 'heuristic':
            all_agent_processes.append(AgentProcess(HeuristicMeetingAgent, **basic_kwargs))
        elif agent_type == 'llm':
            all_agent_processes.append(AgentProcess(LLMMeetingAgent, **basic_kwargs, **llm_kwargs))
        else:
            raise NotImplementedError(f"agent type {agent_type} is not supported")
        all_agent_name.append(config['agent_names'][i])
        name2idx[config['agent_names'][i]] = i


    if args.multi_process:
        gs.logger.info("Start agent processes")
        for agent_process in all_agent_processes:
            agent_process.start()
        gs.logger.info("Agent processes started")

    agent_actions = {}
    agent_actions_to_print = {}

    # Simulation loop
    env_dt_sim = 0.
    all_task_end = False
    infos={"time_used_by_step": np.zeros(5, dtype=float), "time_used_by_scene_step": np.zeros(5, dtype=float)}
    while not all_task_end:
        lst_time = time.perf_counter()
        obs_printable = [{k: v for k, v in obs[agent_id].items() if not isinstance(v, np.ndarray) \
                          and k != 'gt_seg_entity_idx_to_info' and not isinstance(v, datetime)} for agent_id in obs]

        # update obs and do action
        extra_obs = {"agent_pos_dict": {env.config["agent_names"][i]: {"place": env.obs[i]['current_place'], "pose": env.config["agent_poses"][i]} for i in range(num_agents)}
        }

        for i, agent in enumerate(all_agent_processes):
            obs[i].update(extra_obs)
            agent.update(obs[i])

        for i, agent in enumerate(all_agent_processes):
            tm = time.time()
            action = agent.act()
            agent_actions[i] = action
            agent_actions_to_print[agent.name] = agent_actions[i]['type'] if agent_actions[i] is not None else None

        steps_info_path = os.path.join(output_dir, "steps.json")
        if os.path.exists(steps_info_path):
            with open(steps_info_path, 'r') as file:
                steps_info = json.load(file)
        else:
            steps_info = {}

        step_info = {"curr_time": env.curr_time.strftime("%H:%M:%S"),
                     "obs": obs_printable,
                     "action": agent_actions,
                     }
        steps_info[env.steps] = step_info
        try:
            with open(steps_info_path, 'w') as file:
                json.dump(steps_info, file, indent=4)
        except Exception as e:
            import pdb; pdb.set_trace()

        gs.logger.info(f"current time: {env.curr_time}, ViCo steps: {env.steps}/{args.step_limit}, agent_pose: {round_numericals(env.config['agent_poses'])}, agents actions: {agent_actions_to_print}")
        dt_agent = time.perf_counter() - lst_time
        env.config["dt_agent"] = (env.config["dt_agent"] * env.steps + dt_agent) / (env.steps + 1)
        lst_time = time.perf_counter()
        obs, _, done, info = env.step(agent_actions)
        dt_sim = time.perf_counter() - lst_time
        env_dt_sim = (env_dt_sim * (env.steps - 1) + dt_sim) / env.steps
        gs.logger.info(f"Time used: {dt_agent:.2f}s for agents and robots, {dt_sim:.2f}s for simulation, "
                       f"average {env.config['dt_agent']:.2f}s for agents, "
                       f"{env_dt_sim:.2f}s for simulation, "
                       f"{env.config['dt_chat']:.2f}s for post-chatting over {env.steps} steps.")
        for key in info:
            infos[key]+=info[key]
        max_distance=0.
        for agent_pose in env.config['agent_poses']:
            agent_pose = np.array(agent_pose[:2])
            for agent2_pose in env.config['agent_poses']:
                agent2_pose = np.array(agent2_pose[:2])
                max_distance = max(max_distance, np.linalg.norm(agent_pose-agent2_pose))
        gs.logger.info(f"The longest distance between the agents: {max_distance:.2f}")

        all_task_end = True
        for agent in agent_actions_to_print:
            action = agent_actions_to_print[agent]
            if (action is None or action != 'task_complete') and env.steps <= 1500:
                all_task_end = False

    result = {"agent_poses": [agent_pose for agent_pose in env.config['agent_poses']],
              "time_spent_meeting": env.steps,
              "done": True}
    with open(result_path, 'w') as file:
        json.dump(result, file, indent=4)
    gs.logger.warning(f"{result}")
    env.close()
    del env
    end_processes()

if __name__ == '__main__':
    keep_running = os.environ.get('keep_running', '0') == '1'
    t = threading.Thread(target=gpu_occupy_loop, daemon=True)
    t.start()
    try:
        main()
    finally:
        keep_running = False
        t.join(timeout=1)