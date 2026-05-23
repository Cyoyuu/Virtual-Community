"""PPO training driver for the RL meeting-challenge baseline.

Runs sequential episode rollouts through ``meeting_challenge.challenge.run_challenge``,
collects per-decision (obs, action, logprob, value, reward) tuples from each
RL agent via ``rl_trajectory.pkl`` files written under the agent storage paths,
and applies a PPO update after every rollout batch.

Usage:
    PYTHONPATH=. python meeting_challenge/train_rl.py \
        --scenes NY PARIS --agent_num 3 --sentinel_type stationary --sentinel_num 1 \
        --total_episodes 200 --save_dir meeting_challenge/checkpoints

    PYTHONPATH=. python meeting_challenge/train_rl.py --debug-single-episode --scenes NY

The trainer uses ``--rl_training_mode`` internally; you do not need to set it.
"""
from __future__ import annotations

import argparse
import copy
import os
import pickle
import random
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

# Make repo root importable when launched as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.meeting_challenge.rl_policy import (
    K,
    PlacePolicy,
    PolicyConfig,
    save_policy,
)


# ---- full training-state checkpointing (resumable) ----------------------------

def _save_training_state(path: str, policy: PlacePolicy, optimizer,
                         ep_count: int, episode_returns: list,
                         batch_samples_count: int, args) -> None:
    """Atomic save of *everything* needed to resume mid-training: policy +
    optimizer + episode counter + rng state + reward history. Used after every
    episode so a SLURM SIGTERM loses at most one episode's compute.
    """
    blob = {
        "state_dict": {k: v.detach().cpu() for k, v in policy.state_dict().items()},
        "optimizer_state": optimizer.state_dict(),
        "ep_count": ep_count,
        "episode_returns": list(episode_returns),
        "batch_samples_count": batch_samples_count,
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(),
        "policy_config": PolicyConfig().__dict__,
        "trainer_args": vars(args),
    }
    tmp = path + ".tmp"
    torch.save(blob, tmp)
    os.replace(tmp, path)


def _load_training_state(path: str, policy: PlacePolicy, optimizer, device: str):
    """Restore training state into `policy` and `optimizer` in place.
    Returns (ep_count, episode_returns)."""
    # Always load to CPU first: torch.set_rng_state() requires a CPU ByteTensor,
    # and Adam optimizer state restore is happier when the tensors match the
    # device of their corresponding parameters (which we set below).
    blob = torch.load(path, map_location="cpu", weights_only=False)
    policy.load_state_dict(blob["state_dict"])
    policy.to(device)
    if "optimizer_state" in blob:
        optimizer.load_state_dict(blob["optimizer_state"])
        # Move optimizer state tensors to the same device as the parameters.
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    if "rng_python" in blob:
        random.setstate(blob["rng_python"])
    if "rng_numpy" in blob:
        np.random.set_state(blob["rng_numpy"])
    if "rng_torch" in blob:
        torch.set_rng_state(blob["rng_torch"].cpu())
    return blob.get("ep_count", 0), blob.get("episode_returns", [])


# ---- trainer CLI --------------------------------------------------------------

def build_trainer_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", nargs="+", default=["NY"],
                   help="Scenes to rotate through (one per episode).")
    p.add_argument("--config", type=str, default="agents_num_15")
    p.add_argument("--agent_num", type=int, default=3)
    p.add_argument("--sentinel_type", type=str, default="stationary",
                   choices=["stationary", "patrol"])
    p.add_argument("--sentinel_num", type=int, default=1)
    p.add_argument("--step_limit", type=int, default=1500)
    p.add_argument("--enable_danger_zone", action="store_true")
    p.add_argument("--gt_only_for_sentinels", action="store_true")
    p.add_argument("--enable_gt_segmentation", action="store_true", default=True)
    p.add_argument("--backend", type=str, default="gpu", choices=["gpu", "cpu"])

    # PPO
    p.add_argument("--total_episodes", type=int, default=200)
    p.add_argument("--batch_episodes", type=int, default=4,
                   help="Episodes per PPO update.")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--epochs_per_update", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=64)

    # Reward
    p.add_argument("--reward_done", type=float, default=1.0)
    p.add_argument("--reward_caught", type=float, default=-1.0)
    p.add_argument("--reward_time_coef", type=float, default=-0.001,
                   help="Multiplied by time_spent_meeting / step_limit.")
    p.add_argument("--reward_shape_dist", type=float, default=0.0,
                   help="Per-decision shaping coefficient on min_dist_to_warning.")

    # I/O
    p.add_argument("--save_dir", type=str, default="meeting_challenge/checkpoints")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to an existing .pt to warm-start from.")
    p.add_argument("--rollout_output_dir", type=str,
                   default="meeting_challenge/output/rl_train",
                   help="Where each training episode's run artifacts land.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--debug-single-episode", action="store_true",
                   help="Run exactly one rollout with a random-init policy "
                        "and one PPO update, then exit. For smoke-testing.")
    return p.parse_args()


# ---- args bridge: trainer args -> challenge.parse_args() shape -----------------

def make_challenge_args(trainer_args, scene: str, job_id: int,
                        ckpt_path: str, output_dir: str):
    """Construct an argparse.Namespace that ``run_challenge`` accepts."""
    return argparse.Namespace(
        # general
        seed=trainer_args.seed,
        precision="32",
        logging_level="info",
        backend=trainer_args.backend,
        head_less=True,
        multi_process=True,
        output_dir=output_dir,
        debug=False,
        overwrite=True,
        job_id=job_id,
        # simulation
        resolution=512,
        enable_collision=False,
        skip_avatar_animation=True,
        enable_gt_segmentation=trainer_args.enable_gt_segmentation,
        max_seconds=86400,
        save_per_seconds=200,
        enable_third_person_cameras=False,
        curr_time=None,
        start_id=None,
        only_one_sample=False,
        use_luisa_renderer=False,
        # scene
        scene=scene,
        enable_indoor_scene=True,
        enable_indoor_activities=False,
        enable_outdoor_objects=True,
        outdoor_objects_assets_dir="scene/object_assets",
        outdoor_objects_max_num=5,
        no_load_scene=False,
        # traffic
        no_traffic_manager=False,
        tm_vehicle_num=0,
        tm_avatar_num=0,
        enable_tm_debug=False,
        # agent
        config=trainer_args.config,
        agent_num=trainer_args.agent_num,
        agent_type="rl",
        agent_type2=None,
        no_react=False,
        lm_source="azure",
        lm_id="gpt-4o",
        max_tokens=4096,
        temperature=0.0,
        top_p=1.0,
        server_port=8000,
        # meeting challenge
        robot_as_agent=False,
        enable_demo_camera=False,
        step_limit=trainer_args.step_limit,
        robot_policy_path="",
        sentinel_type=trainer_args.sentinel_type,
        sentinel_num=trainer_args.sentinel_num,
        enable_danger_zone=trainer_args.enable_danger_zone,
        refine_retry=5,
        gt_only_for_sentinels=trainer_args.gt_only_for_sentinels,
        detect_interval=-1,
        ablate="",
        replay_mode=False,
        rl_ckpt=ckpt_path,
        rl_training_mode=True,
    )


# ---- reward + GAE -------------------------------------------------------------

def compute_episode_reward(result: dict, banned_agents: list, agent_name: str,
                           args) -> float:
    """Terminal scalar reward for a given agent based on episode outcome."""
    r = 0.0
    if result.get("done", 0) == 1:
        r += args.reward_done
    if agent_name in banned_agents:
        r += args.reward_caught
    r += args.reward_time_coef * (result.get("time_spent_meeting", 0) /
                                  float(args.step_limit))
    return r


def compute_gae(rewards: List[float], values: List[float], gamma: float,
                lam: float) -> List[float]:
    """Standard GAE on a flat trajectory; bootstrap value = 0 (terminal)."""
    advantages = [0.0] * len(rewards)
    gae = 0.0
    next_value = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
        next_value = values[t]
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


# ---- trajectory loading -------------------------------------------------------

@dataclass
class DecisionSample:
    g: torch.Tensor
    c: torch.Tensor
    m: torch.Tensor
    action: int
    old_logprob: float
    value: float
    advantage: float
    return_: float


def load_episode_samples(ep_info: dict, args) -> List[DecisionSample]:
    """Load trajectories from each RL agent's storage path, assemble samples."""
    config_path = ep_info["config_path"]
    agent_names = ep_info["agent_names"]
    result = ep_info["result"]
    banned = ep_info.get("banned_agent_list", [])

    all_samples: List[DecisionSample] = []
    for name in agent_names:
        traj_path = os.path.join(config_path, name, "rl_trajectory.pkl")
        if not os.path.exists(traj_path):
            continue
        with open(traj_path, "rb") as f:
            traj = pickle.load(f)
        if not traj:
            continue
        rewards = [0.0] * len(traj)
        rewards[-1] = compute_episode_reward(result, banned, name, args)
        values = [t["value"] for t in traj]
        advantages, returns = compute_gae(rewards, values, args.gamma,
                                          args.gae_lambda)
        for t, adv, ret in zip(traj, advantages, returns):
            all_samples.append(DecisionSample(
                g=t["global"], c=t["cand"], m=t["mask"],
                action=t["action"], old_logprob=t["logprob"],
                value=t["value"], advantage=adv, return_=ret,
            ))
    return all_samples


# ---- PPO update ---------------------------------------------------------------

def ppo_update(policy: PlacePolicy, optimizer, samples: List[DecisionSample],
               args, device: str):
    if not samples:
        return {"loss": 0.0, "n": 0}
    g = torch.stack([s.g for s in samples]).to(device)
    c = torch.stack([s.c for s in samples]).to(device)
    m = torch.stack([s.m for s in samples]).to(device)
    actions = torch.tensor([s.action for s in samples], dtype=torch.long,
                           device=device)
    old_lp = torch.tensor([s.old_logprob for s in samples], dtype=torch.float32,
                          device=device)
    adv = torch.tensor([s.advantage for s in samples], dtype=torch.float32,
                       device=device)
    ret = torch.tensor([s.return_ for s in samples], dtype=torch.float32,
                       device=device)
    # Normalize advantages.
    if adv.numel() > 1:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    N = g.shape[0]
    idxs = np.arange(N)
    total_loss = 0.0
    for _ in range(args.epochs_per_update):
        np.random.shuffle(idxs)
        for start in range(0, N, args.minibatch):
            mb = idxs[start:start + args.minibatch]
            logits, value = policy(g[mb], c[mb], m[mb])
            dist = torch.distributions.Categorical(logits=logits)
            new_lp = dist.log_prob(actions[mb])
            entropy = dist.entropy().mean()
            ratio = torch.exp(new_lp - old_lp[mb])
            unclipped = ratio * adv[mb]
            clipped = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv[mb]
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = F.mse_loss(value, ret[mb])
            loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()
            total_loss += float(loss.item())
    return {"loss": total_loss, "n": N}


# ---- main loop ----------------------------------------------------------------

def main():
    args = build_trainer_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    policy = PlacePolicy().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    # Two checkpoint files live in save_dir:
    #   latest.pt    -- policy-only blob (read by rollout agents, on CPU)
    #   training_state.pt -- full resume state (policy + optimizer + ep counter + RNG)
    latest_path = os.path.join(args.save_dir, "latest.pt")
    training_state_path = os.path.join(args.save_dir, "training_state.pt")

    start_ep = 0
    episode_returns: List[float] = []
    # If --resume is set, prefer it; otherwise auto-resume from training_state.pt
    # if it exists. This lets a SLURM requeue just re-run the same command.
    resume_path = args.resume
    if resume_path is None and os.path.exists(training_state_path):
        resume_path = training_state_path
    if resume_path is not None and os.path.exists(resume_path):
        start_ep, episode_returns = _load_training_state(
            resume_path, policy, optimizer, device
        )
        print(f"[train_rl] resumed from {resume_path} at ep {start_ep}")
    else:
        print(f"[train_rl] starting fresh (no checkpoint at {training_state_path})")

    # Flush the rollout-facing latest.pt so the first episode picks up the
    # (possibly resumed) weights.
    save_policy(policy.cpu(), PolicyConfig(), latest_path)
    policy.to(device)

    # Import run_challenge lazily so the trainer can import torch first
    # without genesis side effects when only doing offline ops.
    from meeting_challenge.challenge import run_challenge

    total = (start_ep + 1) if args.debug_single_episode else args.total_episodes
    if start_ep >= total:
        print(f"[train_rl] already at ep {start_ep} >= total {total}; nothing to do")
        return
    batch_samples: List[DecisionSample] = []
    for ep in range(start_ep, total):
        scene = args.scenes[ep % len(args.scenes)]
        ep_output_dir = os.path.join(args.rollout_output_dir,
                                     f"ep{ep:05d}_{scene}")
        if os.path.exists(ep_output_dir):
            shutil.rmtree(ep_output_dir, ignore_errors=True)
        os.makedirs(ep_output_dir, exist_ok=True)
        # Always use job_id=0 for training rollouts. challenge.py writes a
        # second copy of result.json to meeting_challenge/results_<gt>/.../
        # using job_id, and cal_results.py only aggregates job_ids 1..6 -- so
        # job_0 is the "reserved for training" slot that eval ignores. The
        # rollout's own output_dir is unique per episode (`ep_output_dir`),
        # so trajectory pickles still don't collide across episodes.
        ch_args = make_challenge_args(args, scene=scene, job_id=0,
                                      ckpt_path=latest_path,
                                      output_dir=ep_output_dir)
        print(f"[train_rl] ep {ep}/{total} scene={scene} starting ...")
        t0 = time.time()
        ep_info = run_challenge(ch_args)
        wall = time.time() - t0
        result = ep_info["result"]
        samples = load_episode_samples(ep_info, args)
        # episode-level scalar return for logging.
        ep_return = compute_episode_reward(
            result, ep_info.get("banned_agent_list", []),
            agent_name=(ep_info["agent_names"][0] if ep_info["agent_names"] else ""),
            args=args,
        )
        episode_returns.append(ep_return)
        batch_samples.extend(samples)
        print(f"[train_rl] ep {ep} done={result.get('done')} "
              f"caught={result.get('caught_rate'):.2f} "
              f"t={result.get('time_spent_meeting')} "
              f"R={ep_return:.3f} samples={len(samples)} wall={wall:.1f}s")

        if (ep + 1) % args.batch_episodes == 0 or ep == total - 1:
            stats = ppo_update(policy, optimizer, batch_samples, args, device)
            print(f"[train_rl] PPO update on {stats['n']} samples, "
                  f"loss_total={stats['loss']:.3f}, "
                  f"avg_return={np.mean(episode_returns[-args.batch_episodes:]):.3f}")
            batch_samples = []
            # Flush new policy weights for the next rollout to pick up.
            save_policy(policy.cpu(), PolicyConfig(), latest_path)
            policy.to(device)
            ckpt_path = os.path.join(args.save_dir, f"rl_ep{ep:05d}.pt")
            save_policy(policy.cpu(), PolicyConfig(), ckpt_path)
            policy.to(device)

        # Persist full training state at the end of every episode, regardless
        # of PPO update cadence. This caps the recovery cost at one episode if
        # the SLURM job is killed mid-training. ``ep + 1`` is the next episode
        # to run on resume.
        _save_training_state(
            training_state_path, policy.cpu(), optimizer,
            ep_count=ep + 1, episode_returns=episode_returns,
            batch_samples_count=len(batch_samples), args=args,
        )
        policy.to(device)


if __name__ == "__main__":
    main()
