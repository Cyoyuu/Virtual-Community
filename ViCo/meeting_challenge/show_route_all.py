import os
import json
import argparse
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle


def load_steps(main_steps_json):
    with open(main_steps_json, "r") as f:
        data = json.load(f)
    step_ids = sorted([int(k) for k in data.keys()])
    steps_data = {k: data[str(k)]["obs"] for k in step_ids}
    return step_ids, steps_data


def load_bus_pose(env_dir, step):
    env_path = os.path.join(env_dir, f"{step:06d}.json")
    if not os.path.isfile(env_path):
        return None
    try:
        with open(env_path, "r") as f:
            env = json.load(f)
        pose = env.get("curr_bus_pose", None)
        if isinstance(pose, (list, tuple)) and len(pose) >= 2:
            return float(pose[0]), float(pose[1])
    except Exception:
        pass
    return None


def is_sentinel(name: str) -> bool:
    return isinstance(name, str) and name.startswith("Sentinel")


def build_entity_lists(steps_data, track_sentinels=False):
    agents = set()
    sentinels = set()
    for obs_list in steps_data.values():
        for obs in obs_list:
            name = obs.get("name")
            if not name:
                continue
            if is_sentinel(name):
                sentinels.add(name)
            else:
                agents.add(name)
    if not track_sentinels:
        return sorted(list(agents)), sorted(list(sentinels))
    else:
        return sorted(list(agents)), sorted(list(sentinels))


def precompute_histories(full_steps, steps_data, agents, max_jump=100.0):
    idx_of_full = {s: i for i, s in enumerate(full_steps)}
    agent_traces = {name: {"xs": [None] * len(full_steps), "ys": [None] * len(full_steps)} for name in agents}

    for step, obs_list in steps_data.items():
        if step not in idx_of_full:
            continue
        i = idx_of_full[step]
        for obs in obs_list:
            name = obs.get("name")
            if name in agent_traces:
                pose = obs.get("pose", None)
                if isinstance(pose, (list, tuple)) and len(pose) >= 2:
                    agent_traces[name]["xs"][i] = float(pose[0])
                    agent_traces[name]["ys"][i] = float(pose[1])

    for name, trace in agent_traces.items():
        xs, ys = trace["xs"], trace["ys"]
        last_x, last_y = None, None
        for i in range(len(xs)):
            cur_x, cur_y = xs[i], ys[i]

            if cur_x is None or cur_y is None:
                xs[i], ys[i] = last_x, last_y
                continue

            if last_x is not None and last_y is not None:
                dx = cur_x - last_x
                dy = cur_y - last_y
                dist = (dx * dx + dy * dy) ** 0.5
                if dist > max_jump:
                    xs[i], ys[i] = last_x, last_y
                    continue

            last_x, last_y = xs[i], ys[i]

    return agent_traces, idx_of_full


def animate_all(scene, max_steps=2000, interval=100, fps=5, save_gif=True):
    base_dir = f"ViCo/meeting_challenge/output/{scene}/heuristic_nav"
    steps_json = os.path.join(base_dir, "steps.json")
    env_dir = os.path.join(base_dir, "steps", "env")

    if not os.path.isfile(steps_json):
        raise FileNotFoundError(f"steps.json not found at: {steps_json}")

    step_ids, steps_data = load_steps(steps_json)
    if len(step_ids) == 0:
        raise RuntimeError("steps.json contains no steps.")

    if len(step_ids) >= 2:
        start_step = step_ids[1]
    else:
        start_step = step_ids[0]

    end_step = max_steps
    full_steps = list(range(start_step, end_step + 1))
    frames = full_steps[::5]
    
    agents, sentinels = build_entity_lists(steps_data, track_sentinels=False)

    agent_traces, idx_of_full = precompute_histories(full_steps, steps_data, agents, max_jump=100.0)

    bg_path = f"ViCo/assets/scenes/{scene}/global.png"
    if not os.path.isfile(bg_path):
        raise FileNotFoundError(f"Background map not found at: {bg_path}")
    bg = mpimg.imread(bg_path)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(bg, extent=[-512, 512, -512, 512], zorder=0, alpha=0.4)
    ax.set_xlim(-400, 400)
    ax.set_ylim(-400, 400)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.grid(True, linestyle="--", alpha=0.1)

    bright_palette = [
        "#0E7EEE",
        "#E29F22",
        "#0DE50D",
        "#C00B0B",
        "#9106CD",
    ]
    agent_lines = {}
    agent_dots = {}
    for i, name in enumerate(agents):
        color = bright_palette[i % len(bright_palette)]
        (line,) = ax.plot([], [], linewidth=2.5, color=color, label=f"{name} path", zorder=2)
        dot = ax.scatter([], [], s=18, marker="o", color=color, zorder=3)
        agent_lines[name] = line
        agent_dots[name] = dot

    # --- Sentinel as oriented triangles + following circles ---
    sentinel_last_pos = {}
    sentinel_prev_pos = {}
    sentinel_angles = {}
    sentinel_tris = {}
    sentinel_circles = {}

    def make_triangle_artist():
        (tri_line,) = ax.plot([], [], linestyle="None",
                              marker=(3, 0, 0),
                              markersize=12,
                              markerfacecolor="black",
                              markeredgecolor="white",
                              markeredgewidth=0.8,
                              zorder=4)
        return tri_line

    def make_circle_patch():
        circ = Circle((0, 0),
                      radius=15,
                      facecolor="#FFF3B0",
                      edgecolor="none",
                      alpha=0.4,
                      zorder=3.6)
        ax.add_patch(circ)
        circ.set_visible(False)
        return circ

    for sname in sentinels:
        sentinel_tris[sname] = make_triangle_artist()
        sentinel_angles[sname] = 0.0
        sentinel_circles[sname] = make_circle_patch()

    bus_scat = ax.scatter([], [], s=55, marker="s", color="red", label="Bus (now)", zorder=5)

    sentinel_legend_proxy, = ax.plot([], [], linestyle="None",
                                     marker=(3, 0, 0),
                                     markersize=12,
                                     markerfacecolor="black",
                                     markeredgecolor="white",
                                     markeredgewidth=0.8,
                                     label="Sentinel (heading)",
                                     zorder=4)
    handles = list(agent_lines.values())
    if len(handles) > 0:
        handles = handles[:1]
        handles[0].set_label("Agent paths & heads")
    handles += [sentinel_legend_proxy, bus_scat]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.8)
    ax.set_title(f"{scene}: Agents history & Sentinels/Bus live positions")

    def get_obs_by_name_exact(step, target_name):
        lst = steps_data.get(step, [])
        for obs in lst:
            if obs.get("name") == target_name:
                pose = obs.get("pose", None)
                if isinstance(pose, (list, tuple)) and len(pose) >= 2:
                    return float(pose[0]), float(pose[1])
        return None

    def heading_from(prev_xy, curr_xy, default_deg):
        if prev_xy is None or curr_xy is None:
            return default_deg
        px, py = prev_xy
        cx, cy = curr_xy
        dx, dy = cx - px, cy - py
        if dx == 0 and dy == 0:
            return default_deg
        theta = np.degrees(np.arctan2(dy, dx)) - 90.0
        return float(theta)

    def update(step):
        ax.set_title(f"{scene}: Agents history & Sentinels/Bus live positions — step {step}")

        idx = idx_of_full.get(step, len(full_steps) - 1)
        # Agents
        for name in agents:
            xs = agent_traces[name]["xs"][: idx + 1]
            ys = agent_traces[name]["ys"][: idx + 1]
            agent_lines[name].set_data(xs, ys)
            cx, cy = agent_traces[name]["xs"][idx], agent_traces[name]["ys"][idx]
            if cx is not None and cy is not None:
                agent_dots[name].set_offsets(np.array([[cx, cy]]))
            else:
                agent_dots[name].set_offsets(np.empty((0, 2)))

        # Sentinels
        for sname in sentinels:
            pos_now = get_obs_by_name_exact(step, sname)
            if pos_now is not None:
                prev = sentinel_last_pos.get(sname, None)
                if prev is not None and (pos_now[0] != prev[0] or pos_now[1] != prev[1]):
                    sentinel_prev_pos[sname] = prev
                sentinel_last_pos[sname] = pos_now

            eff_pos = sentinel_last_pos.get(sname, None)
            tri = sentinel_tris[sname]
            circ = sentinel_circles[sname]

            if eff_pos is not None:
                circ.center = eff_pos
                circ.set_visible(True)

                default_ang = sentinel_angles.get(sname, 0.0)
                ang = heading_from(sentinel_prev_pos.get(sname, None), eff_pos, default_ang)
                sentinel_angles[sname] = ang

                tri.set_data([eff_pos[0]], [eff_pos[1]])
                tri.set_marker((3, 0, ang))
            else:
                circ.set_visible(False)
                tri.set_data([], [])
                tri.set_marker((3, 0, sentinel_angles.get(sname, 0.0)))

        bus_pos = load_bus_pose(env_dir, step)
        if bus_pos is not None:
            bus_scat.set_offsets(np.array([[bus_pos[0], bus_pos[1]]]))
        else:
            bus_scat.set_offsets(np.empty((0, 2)))

        artists = (
            list(agent_lines.values())
            + list(agent_dots.values())
            + list(sentinel_tris.values())
            + list(sentinel_circles.values())
            + [bus_scat]
        )
        return artists

    ani = FuncAnimation(fig, update, frames=frames, interval=interval, blit=False, repeat=False)

    if save_gif:
        out_path = f"{scene}_history_all.gif"
        ani.save(out_path, writer="pillow", fps=fps)
        print(f"Saved animation to {out_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", "-s", type=str, required=True, help="Scene name, e.g., HARVARD")
    parser.add_argument("--time", "-t", type=int, default=200, help="Max steps to animate (inclusive)")
    parser.add_argument("--interval", "-i", type=int, default=100, help="Delay between frames (ms)")
    parser.add_argument("--fps", type=int, default=5, help="FPS when saving GIF")
    parser.add_argument("--save_gif", action="store_true", help="Save to GIF instead of interactive show()")
    args = parser.parse_args()

    animate_all(scene=args.scene, max_steps=args.time, interval=args.interval, fps=args.fps, save_gif=args.save_gif)


if __name__ == "__main__":
    main()