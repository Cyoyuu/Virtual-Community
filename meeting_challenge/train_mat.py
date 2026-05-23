"""PPO training driver for the MAT meeting-challenge baseline.

Differences from train_rl.py:
- Each macro decision is a *team* record (one global state, N joint actions,
  one team value) rather than per-agent independent records.
- Loss treats the joint policy probability as the product over agents; under
  the multi-agent advantage decomposition theorem this is the right surrogate
  for HAPPO. We use plain PPO clipping on the joint ratio for simplicity.
- The MAT trajectory is written once per episode by the in-process
  ``MATController`` to ``<output_dir>/mat_trajectory.pkl``.
"""
from __future__ import annotations

import argparse
import os
import pickle
import random
import shutil
import sys
import time
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.meeting_challenge.mat_policy import (
    K, MATConfig, MATPolicy, save_policy,
)


# ---- training-state checkpointing --------------------------------------------

def _save_training_state(path, policy, optimizer, ep_count, episode_returns, args):
    blob = {
        "state_dict": {k: v.detach().cpu() for k, v in policy.state_dict().items()},
        "optimizer_state": optimizer.state_dict(),
        "ep_count": ep_count,
        "episode_returns": list(episode_returns),
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(),
        "config": MATConfig(num_agents=args.agent_num).__dict__,
        "trainer_args": vars(args),
    }
    tmp = path + ".tmp"
    torch.save(blob, tmp)
    os.replace(tmp, path)


def _load_training_state(path, policy, optimizer, device):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    policy.load_state_dict(blob["state_dict"])
    policy.to(device)
    if "optimizer_state" in blob:
        optimizer.load_state_dict(blob["optimizer_state"])
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


# ---- CLI ----------------------------------------------------------------------

def build_trainer_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", nargs="+", default=["NY"])
    p.add_argument("--config", type=str, default="agents_num_15")
    p.add_argument("--agent_num", type=int, default=5)
    p.add_argument("--sentinel_type", type=str, default="stationary",
                   choices=["stationary", "patrol"])
    p.add_argument("--sentinel_num", type=int, default=10)
    p.add_argument("--step_limit", type=int, default=1500)
    p.add_argument("--planning_interval", type=int, default=50,
                   help="Macro-action duration; matches CoELA default 50.")
    p.add_argument("--enable_danger_zone", action="store_true")
    p.add_argument("--gt_only_for_sentinels", action="store_true")
    p.add_argument("--enable_gt_segmentation", action="store_true", default=True)
    p.add_argument("--backend", type=str, default="gpu", choices=["gpu", "cpu"])

    # PPO (CoELA reports: hidden=64, lr=7e-4, ppo_epoch=10)
    p.add_argument("--total_episodes", type=int, default=200)
    p.add_argument("--batch_episodes", type=int, default=4)
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--epochs_per_update", type=int, default=10)
    p.add_argument("--minibatch", type=int, default=32)

    # Reward (same shape as train_rl.py)
    p.add_argument("--reward_done", type=float, default=1.0)
    p.add_argument("--reward_caught", type=float, default=-1.0)
    p.add_argument("--reward_time_coef", type=float, default=-0.001)

    p.add_argument("--save_dir", type=str, default="meeting_challenge/checkpoints_mat")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--rollout_output_dir", type=str,
                   default="meeting_challenge/output/mat_train")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--debug-single-episode", action="store_true")
    return p.parse_args()


# ---- challenge.py args bridge -------------------------------------------------

def make_challenge_args(trainer_args, scene, job_id, ckpt_path, output_dir):
    return argparse.Namespace(
        seed=trainer_args.seed, precision="32", logging_level="info",
        backend=trainer_args.backend, head_less=True,
        multi_process=False,            # MAT requires in-process agents
        output_dir=output_dir, debug=False, overwrite=True, job_id=job_id,
        resolution=512, enable_collision=False, skip_avatar_animation=True,
        enable_gt_segmentation=trainer_args.enable_gt_segmentation,
        max_seconds=86400, save_per_seconds=200,
        enable_third_person_cameras=False, curr_time=None, start_id=None,
        only_one_sample=False, use_luisa_renderer=False,
        scene=scene,
        enable_indoor_scene=True, enable_indoor_activities=False,
        enable_outdoor_objects=True,
        outdoor_objects_assets_dir="scene/object_assets",
        outdoor_objects_max_num=5, no_load_scene=False,
        no_traffic_manager=False, tm_vehicle_num=0, tm_avatar_num=0,
        enable_tm_debug=False,
        config=trainer_args.config, agent_num=trainer_args.agent_num,
        agent_type="mat", agent_type2=None, no_react=False,
        lm_source="azure", lm_id="gpt-4o", max_tokens=4096,
        temperature=0.0, top_p=1.0, server_port=8000,
        robot_as_agent=False, enable_demo_camera=False,
        step_limit=trainer_args.step_limit, robot_policy_path="",
        sentinel_type=trainer_args.sentinel_type,
        sentinel_num=trainer_args.sentinel_num,
        enable_danger_zone=trainer_args.enable_danger_zone,
        refine_retry=5,
        gt_only_for_sentinels=trainer_args.gt_only_for_sentinels,
        detect_interval=-1, ablate="", replay_mode=False,
        rl_ckpt=None, rl_training_mode=False,
        mat_ckpt=ckpt_path, mat_training_mode=True,
        mat_planning_interval=trainer_args.planning_interval,
    )


# ---- reward + GAE -------------------------------------------------------------

def compute_team_reward(result, banned_agents, args):
    r = 0.0
    if result.get("done", 0) == 1:
        r += args.reward_done
    if banned_agents:
        # Per banned agent, dock proportional to caught_rate.
        r += args.reward_caught * (len(banned_agents) / args.agent_num)
    r += args.reward_time_coef * (result.get("time_spent_meeting", 0) /
                                  float(args.step_limit))
    return r


def compute_gae(rewards, values, gamma, lam):
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


@dataclass
class TeamSample:
    gmap: torch.Tensor          # (GC, H, W)
    smaps: torch.Tensor         # (N, 1, H, W)
    feats: torch.Tensor         # (N, F)
    mask: torch.Tensor          # (K,)
    actions: torch.Tensor       # (N,)
    old_logprob_sum: float      # sum of per-agent log probs (joint)
    value: float
    advantage: float = 0.0
    return_: float = 0.0


def load_episode_samples(ep_info, args) -> List[TeamSample]:
    """Read mat_trajectory.pkl and assign the terminal team reward."""
    traj_path = ep_info.get("mat_trajectory_path")
    if traj_path is None or not os.path.exists(traj_path):
        return []
    with open(traj_path, "rb") as f:
        traj = pickle.load(f)
    if not traj:
        return []
    result = ep_info["result"]
    banned = ep_info.get("banned_agent_list", [])
    rewards = [0.0] * len(traj)
    rewards[-1] = compute_team_reward(result, banned, args)
    values = [t["value"] for t in traj]
    advs, rets = compute_gae(rewards, values, args.gamma, args.gae_lambda)

    samples: List[TeamSample] = []
    for t, adv, ret in zip(traj, advs, rets):
        samples.append(TeamSample(
            gmap=t["global_map"].squeeze(0) if t["global_map"].dim() == 4 else t["global_map"],
            smaps=t["self_maps"],
            feats=t["agent_feats"],
            mask=t["candidate_mask"],
            actions=t["actions"].long(),
            old_logprob_sum=float(t["logprob"].sum().item()),
            value=float(t["value"]),
            advantage=float(adv),
            return_=float(ret),
        ))
    return samples


def ppo_update(policy, optimizer, samples, args, device):
    if not samples:
        return {"loss": 0.0, "n": 0}
    gmap = torch.stack([s.gmap for s in samples]).to(device)
    smaps = torch.stack([s.smaps for s in samples]).to(device)
    feats = torch.stack([s.feats for s in samples]).to(device)
    mask = torch.stack([s.mask for s in samples]).to(device)
    actions = torch.stack([s.actions for s in samples]).to(device)
    old_lp = torch.tensor([s.old_logprob_sum for s in samples],
                          dtype=torch.float32, device=device)
    adv = torch.tensor([s.advantage for s in samples],
                       dtype=torch.float32, device=device)
    ret = torch.tensor([s.return_ for s in samples],
                       dtype=torch.float32, device=device)
    if adv.numel() > 1:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    N = gmap.shape[0]
    idxs = np.arange(N)
    total_loss = 0.0
    for _ in range(args.epochs_per_update):
        np.random.shuffle(idxs)
        for start in range(0, N, args.minibatch):
            mb = idxs[start:start + args.minibatch]
            logits, value = policy(
                gmap[mb], smaps[mb], feats[mb], actions[mb], mask[mb],
            )                                                       # (B,N,K), (B,)
            log_probs = torch.log_softmax(logits, dim=-1)           # (B,N,K)
            sel = log_probs.gather(-1, actions[mb].unsqueeze(-1)).squeeze(-1)
            new_lp_sum = sel.sum(dim=-1)                            # (B,)
            ratio = torch.exp(new_lp_sum - old_lp[mb])
            unclipped = ratio * adv[mb]
            clipped = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv[mb]
            policy_loss = -torch.min(unclipped, clipped).mean()
            # Per-agent entropy averaged
            probs = log_probs.exp()
            ent = -(probs * log_probs).sum(dim=-1).mean()
            value_loss = F.mse_loss(value, ret[mb])
            loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * ent
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()
            total_loss += float(loss.item())
    return {"loss": total_loss, "n": N}


# ---- main ---------------------------------------------------------------------

def main():
    args = build_trainer_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    policy = MATPolicy(num_agents=args.agent_num).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    latest_path = os.path.join(args.save_dir, "latest.pt")
    training_state_path = os.path.join(args.save_dir, "training_state.pt")

    start_ep = 0
    episode_returns: List[float] = []
    resume_path = args.resume
    if resume_path is None and os.path.exists(training_state_path):
        resume_path = training_state_path
    if resume_path is not None and os.path.exists(resume_path):
        start_ep, episode_returns = _load_training_state(
            resume_path, policy, optimizer, device)
        print(f"[train_mat] resumed from {resume_path} at ep {start_ep}")
    else:
        print(f"[train_mat] starting fresh (no ckpt at {training_state_path})")

    save_policy(policy.cpu(), MATConfig(num_agents=args.agent_num), latest_path)
    policy.to(device)

    from meeting_challenge.challenge import run_challenge

    total = (start_ep + 1) if args.debug_single_episode else args.total_episodes
    if start_ep >= total:
        print(f"[train_mat] already at ep {start_ep} >= total {total}; done")
        return

    batch_samples: List[TeamSample] = []
    for ep in range(start_ep, total):
        scene = args.scenes[ep % len(args.scenes)]
        ep_output_dir = os.path.join(args.rollout_output_dir,
                                     f"ep{ep:05d}_{scene}")
        if os.path.exists(ep_output_dir):
            shutil.rmtree(ep_output_dir, ignore_errors=True)
        os.makedirs(ep_output_dir, exist_ok=True)
        ch_args = make_challenge_args(args, scene=scene, job_id=0,
                                      ckpt_path=latest_path,
                                      output_dir=ep_output_dir)
        print(f"[train_mat] ep {ep}/{total} scene={scene} ...")
        t0 = time.time()
        ep_info = run_challenge(ch_args)
        wall = time.time() - t0
        result = ep_info["result"]
        samples = load_episode_samples(ep_info, args)
        ep_return = compute_team_reward(
            result, ep_info.get("banned_agent_list", []), args)
        episode_returns.append(ep_return)
        batch_samples.extend(samples)
        print(f"[train_mat] ep {ep} done={result.get('done')} "
              f"caught={result.get('caught_rate'):.2f} "
              f"t={result.get('time_spent_meeting')} "
              f"R={ep_return:.3f} samples={len(samples)} wall={wall:.1f}s")

        if (ep + 1) % args.batch_episodes == 0 or ep == total - 1:
            stats = ppo_update(policy, optimizer, batch_samples, args, device)
            print(f"[train_mat] PPO update on {stats['n']} samples, "
                  f"loss_total={stats['loss']:.3f}, "
                  f"avg_return={np.mean(episode_returns[-args.batch_episodes:]):.3f}")
            batch_samples = []
            save_policy(policy.cpu(),
                        MATConfig(num_agents=args.agent_num), latest_path)
            policy.to(device)
            ckpt_path = os.path.join(args.save_dir, f"mat_ep{ep:05d}.pt")
            save_policy(policy.cpu(),
                        MATConfig(num_agents=args.agent_num), ckpt_path)
            policy.to(device)

        _save_training_state(training_state_path, policy.cpu(), optimizer,
                             ep_count=ep + 1,
                             episode_returns=episode_returns, args=args)
        policy.to(device)


if __name__ == "__main__":
    main()
