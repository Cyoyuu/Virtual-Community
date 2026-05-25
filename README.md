# Sentinel: Embodied Cooperative Spatial Reasoning and Planning

This repo contains code for the following paper:

_Xiangye Lin\*, Hongxin Zhang\*, Ruxi Deng, Qinhong Zhou, Chuang Gan_: Sentinel: Embodied Cooperative Spatial Reasoning and Planning

Paper: _coming soon_

Project Website: _coming soon_

We introduce the **Sentinel Challenge**, a benchmark for studying _Cooperative Spatial Intelligence_, in which multiple decentralized embodied agents must communicate in natural language to agree on a mutually safe and convenient meeting point in city-scale outdoor environments, then navigate there while avoiding dynamic sentinels patrolling the scene with only a coarse map tool for spatial guidance. To address this problem, we propose **CoSaR**, a cooperative spatial reasoning and planning framework that bridges the communication and planning strengths of foundation models with classical spatial navigation algorithms. CoSaR maintains a dynamic spatial memory of poses, ETAs, occupancy, and sentinel danger zones, and uses a spatial-aware reasoning module to decide when to communicate, query the map tool, navigate, or wait. Across 14 city-scale scenes with 3–5 agents, CoSaR consistently achieves faster gathering, shorter path lengths, and improved safety over strong baselines.


## Installation

The Sentinel Challenge is built on top of [Virtual Community](https://github.com/UMass-Embodied-AGI/Virtual-Community). The simulator runs on Python 3.11, CUDA 11.7, Ubuntu 24.04.

```bash
git submodule update --init
conda env create -f env.yaml
conda activate cosar

# Genesis physics engine (submodule)
cd Genesis && pip install -e . && cd ..

# Scene-graph builder (used by the perception module)
cd agents/sg && ./setup.sh && cd ../..
```

### Assets

Download the Virtual Community assets from [Google Drive](https://drive.google.com/drive/u/2/folders/15XR80efNfgdpYi-5dXh3lJ35p9WBqFc5) and organize them under `Genesis/genesis/assets/ViCo/`:

```
Genesis/genesis/assets/ViCo/
├── scene/
├──── v1/
├──── commercial_scenes/ (optional, from GRUtopia)
├── robots/
├── objects/
├── avatars/
└── cars/
```

If you want indoor scenes from GRUtopia, follow [their instructions](https://github.com/OpenRobotLab/GRUtopia?tab=readme-ov-file#%EF%B8%8F-assets) to download `commercial_scenes.zip`; otherwise pass `--no_load_indoor_scene` to the runners.

### System Requirements

- **RAM**: 24 GB minimum / 32 GB recommended
- **VRAM**: 10 GB minimum / 16 GB recommended
- **Disk**: 60 GB minimum / 100 GB recommended


## Run Experiments

The main implementation of _CoSaR_ is in `agents/meeting_challenge/CoSaR.py` (agent), `agents/meeting_challenge/base_nav.py` (shared navigation base), and `agents/meeting_challenge/meeting_prompts/cosar_prompts/` (LLM prompts). The challenge entry point is `meeting_challenge/challenge.py`.

Runners live under `meeting_challenge/`:

| Folder | Perception |
|---|---|
| `meeting_challenge/gt_scripts/` | Ground-truth segmentation for all agents (`--enable_gt_segmentation`) |
| `meeting_challenge/no_gt_scripts/` | Ground-truth segmentation restricted to sentinels (`--gt_only_for_sentinels`); other agents use learned perception |

Each folder contains one script per method (`cosar` = our method, plus baselines `mcts`, `roco`, `coela`, `center`, `center_no_avoidance`, `fixed`, `rl`, `mat`, and CoSaR ablations `_no_analyzer`, `_no_emergency_avoidance`, `_no_refine`, `_no_spatial_memory`, `_qwen`).

For example, to run CoSaR with 3 agents and 1 stationary sentinel in New York under the oracle-perception setting:

```bash
bash meeting_challenge/gt_scripts/run_cosar.sh DETROIT 3 stationary 1 0
```

Same configuration without oracle perception:

```bash
bash meeting_challenge/no_gt_scripts/run_cosar.sh DETROIT 3 stationary 1 0
```


## Citation

If you find our work useful, please consider citing:

```bibtex
@misc{lin2026sentinel,
  title  = {Sentinel: Embodied Cooperative Spatial Reasoning and Planning},
  author = {Lin, Xiangye and Zhang, Hongxin and Deng, Ruxi and Zhou, Qinhong and Gan, Chuang},
  year   = {2026},
}
```


## Acknowledgement

The Sentinel Challenge is built on the [Virtual Community](https://github.com/UMass-Embodied-AGI/Virtual-Community) platform; we thank its authors and the underlying open-source projects it depends on, including [Genesis](https://github.com/Genesis-Embodied-AI/Genesis), [GRUtopia](https://github.com/OpenRobotLab/GRUtopia), [Google 3D Tiles](https://3d-tiles.web.app/), [OpenStreetMap](https://www.openstreetmap.org/), and [Blender](https://www.blender.org/). Our baselines build on [CoELA](https://github.com/UMass-Embodied-AGI/CoELA), [RoCo](https://github.com/MandiZhao/robot-collab), and the [Multi-Agent Transformer](https://github.com/PKU-MARL/Multi-Agent-Transformer).
