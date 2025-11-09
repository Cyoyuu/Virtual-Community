import os
import json
import argparse
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.patches import Wedge

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

def animate_all(scene, max_steps=2000, interval=100, fps=5, out_format="mp4", outfile=None, save_gif_flag=False):
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
    agents_all, sentinels = build_entity_lists(steps_data, track_sentinels=False)
    fixed_agents = agents_all[:5]
    fixed_palette = ["#007BFF", "#00C853", "#FFD600", "#8E44AD", "#FF1744"]
    agent_color = {name: fixed_palette[i] for i, name in enumerate(fixed_agents)}
    agent_traces, idx_of_full = precompute_histories(full_steps, steps_data, fixed_agents, max_jump=100.0)
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
    agent_lines = {}
    agent_dots = {}
    for name in fixed_agents:
        color = agent_color[name]
        (line,) = ax.plot([], [], linewidth=2.5, color=color, label=name, zorder=2)
        dot = ax.scatter([], [], s=18, marker="o", color=color, zorder=3)
        agent_lines[name] = line
        agent_dots[name] = dot
    sentinel_last_pos = {}
    sentinel_prev_pos = {}
    sentinel_angles = {}
    sentinel_tris = {}
    sentinel_wedges = {}
    def make_triangle_artist():
        (tri_line,) = ax.plot([], [], linestyle="None",
                              marker=(3, 0, 0),
                              markersize=12,
                              markerfacecolor="black",
                              markeredgecolor="white",
                              markeredgewidth=0.8,
                              zorder=4)
        return tri_line
    def make_wedge_patch():
        wedge = Wedge(center=(0, 0),
                      r=100,
                      theta1=0,
                      theta2=90,
                      facecolor="#FFF3B0",
                      edgecolor="none",
                      alpha=0.9,
                      zorder=1.0)
        ax.add_patch(wedge)
        wedge.set_visible(False)
        return wedge
    for sname in sentinels:
        sentinel_tris[sname] = make_triangle_artist()
        sentinel_angles[sname] = 0.0
        sentinel_wedges[sname] = make_wedge_patch()
    bus_scat = ax.scatter([], [], s=55, marker="s", color="red", label="Bus (now)", zorder=5)
    sentinel_legend_proxy, = ax.plot([], [], linestyle="None",
                                     marker=(3, 0, 0),
                                     markersize=12,
                                     markerfacecolor="black",
                                     markeredgecolor="white",
                                     markeredgewidth=0.8,
                                     label="Sentinel (heading)",
                                     zorder=4)
    handles = list(agent_lines.values()) + [sentinel_legend_proxy, bus_scat]
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
        dx = cx - px
        dy = cy - py
        if dx == 0 and dy == 0:
            return default_deg
        theta = np.degrees(np.arctan2(dy, dx)) - 90.0
        return float(theta)
    def norm_deg(a):
        a = a % 360.0
        if a < 0:
            a += 360.0
        return a
    def in_sector(cx, cy, r, t1, t2, x, y):
        ang = np.degrees(np.arctan2(y - cy, x - cx))
        a = norm_deg(ang)
        t1n = norm_deg(t1)
        t2n = norm_deg(t2)
        if t1n <= t2n:
            ok_ang = (a >= t1n) and (a <= t2n)
        else:
            ok_ang = (a >= t1n) or (a <= t2n)
        dx = x - cx
        dy = y - cy
        d = np.hypot(dx, dy)
        return ok_ang and (d <= r + 1e-9)
    def agent_walking(name, idx):
        if idx <= 0:
            return False
        xs = agent_traces[name]["xs"]
        ys = agent_traces[name]["ys"]
        x0, y0 = xs[idx - 1], ys[idx - 1]
        x1, y1 = xs[idx], ys[idx]
        if x0 is None or y0 is None or x1 is None or y1 is None:
            return False
        return (abs(x1 - x0) + abs(y1 - y0)) > 1e-6
    def update(step):
        ax.set_title(f"{scene}: Agents history & Sentinels/Bus live positions — step {step}")
        idx = idx_of_full.get(step, len(full_steps) - 1)
        for name in fixed_agents:
            xs = agent_traces[name]["xs"][: idx + 1]
            ys = agent_traces[name]["ys"][: idx + 1]
            agent_lines[name].set_data(xs, ys)
            cx, cy = agent_traces[name]["xs"][idx], agent_traces[name]["ys"][idx]
            if cx is not None and cy is not None:
                agent_dots[name].set_offsets(np.array([[cx, cy]]))
            else:
                agent_dots[name].set_offsets(np.empty((0, 2)))
        for sname in sentinels:
            pos_now = get_obs_by_name_exact(step, sname)
            if pos_now is not None:
                prev = sentinel_last_pos.get(sname, None)
                if prev is not None and (pos_now[0] != prev[0] or pos_now[1] != prev[1]):
                    sentinel_prev_pos[sname] = prev
                sentinel_last_pos[sname] = pos_now
            eff_pos = sentinel_last_pos.get(sname, None)
            tri = sentinel_tris[sname]
            wedge = sentinel_wedges[sname]
            if eff_pos is not None:
                wedge.set_center((eff_pos[0], eff_pos[1]))
                wedge.set_visible(True)
                default_ang = sentinel_angles.get(sname, 0.0)
                ang = heading_from(sentinel_prev_pos.get(sname, None), eff_pos, default_ang)
                sentinel_angles[sname] = ang
                theta_mid = ang + 90.0
                theta1 = theta_mid - 45.0
                theta2 = theta_mid + 45.0
                wedge.set_theta1(theta1)
                wedge.set_theta2(theta2)
                tri.set_data([eff_pos[0]], [eff_pos[1]])
                tri.set_marker((3, 0, ang))
                alert = False
                r = wedge.r if hasattr(wedge, "r") else 20.0
                for aname in fixed_agents:
                    if not agent_walking(aname, idx):
                        continue
                    axc = agent_traces[aname]["xs"][idx]
                    ayc = agent_traces[aname]["ys"][idx]
                    if axc is None or ayc is None:
                        continue
                    if in_sector(eff_pos[0], eff_pos[1], r, theta1, theta2, axc, ayc):
                        alert = True
                        break
                if alert:
                    wedge.set_facecolor("red")
                else:
                    wedge.set_facecolor("#FFF3B0")
            else:
                wedge.set_visible(False)
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
            + list(sentinel_wedges.values())
            + [bus_scat]
        )
        return artists
    ani = FuncAnimation(fig, update, frames=frames, interval=interval, blit=False, repeat=False)
    if save_gif_flag:
        out_format = "gif"
    if out_format.lower() == "mp4":
        out_path = outfile if outfile else f"{scene}_history_all.mp4"
        last_err = None
        for codec in ["libx264", "mpeg4"]:
            try:
                writer = FFMpegWriter(fps=fps, codec=codec)
                ani.save(out_path, writer=writer)
                print(f"Saved animation to {out_path} using codec={codec}")
                break
            except Exception as e:
                last_err = e
                continue
        else:
            raise last_err if last_err else RuntimeError("FFMpegWriter failed with all codecs.")
    elif out_format.lower() == "gif":
        out_path = outfile if outfile else f"{scene}_history_all.gif"
        writer = PillowWriter(fps=fps)
        ani.save(out_path, writer=writer)
        print(f"Saved animation to {out_path}")
    else:
        raise ValueError("Unsupported format. Use 'mp4' or 'gif'.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", "-s", type=str, required=True)
    parser.add_argument("--time", "-t", type=int, default=200)
    parser.add_argument("--interval", "-i", type=int, default=100)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--format", type=str, default="mp4", choices=["mp4", "gif"])
    parser.add_argument("--outfile", type=str, default=None)
    parser.add_argument("--save_gif", action="store_true")
    args = parser.parse_args()
    animate_all(scene=args.scene,
                max_steps=args.time,
                interval=args.interval,
                fps=args.fps,
                out_format=args.format,
                outfile=args.outfile,
                save_gif_flag=args.save_gif)

if __name__ == "__main__":
    main()
