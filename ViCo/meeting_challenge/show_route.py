import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
import os
import numpy as np
from matplotlib.animation import FuncAnimation

def animate_waypoints(route_history, scene, max_steps=2000, interval=10, save_gif=False):
    frames = list(range(0, max_steps + 1, interval))
    # Load background image
    aerial_view = mpimg.imread(f"ViCo/assets/scenes/{scene}/global.png")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(aerial_view, extent=[-512, 512, -512, 512])
    ax.set_xlim(-400, 400)
    ax.set_ylim(-400, 400)
    ax.set_title("Waypoints Path", fontsize=14)
    ax.set_xlabel("X Coordinate", fontsize=12)
    ax.set_ylabel("Y Coordinate", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)

    colors = ['blue', 'red', 'yellow', 'cyan', 'green']
    lines = {}
    scatters = {}

    # Initialize empty line and scatter plots for each agent
    for idx, name in enumerate(route_history):
        (line,) = ax.plot([], [], color=colors[idx % len(colors)],
                          linewidth=2, label=f"Path - {name}", zorder=1)
        scatter = ax.scatter([], [], color='red', s=1, zorder=2,
                             label=f"Waypoints - {name}")
        lines[name] = line
        scatters[name] = scatter

    ax.legend()
    ax.axis('equal')

    def update(step):
        print(step)
        for idx, name in enumerate(route_history):
            if str(step) not in route_history[name]["last_route"]:
                continue
            wps = route_history[name]["last_route"][str(step)]
            xs = [wp["location"][0] for wp in wps]
            ys = [wp["location"][1] for wp in wps]

            lines[name].set_data(xs, ys)
            # coords = np.column_stack((xs, ys)) if xs and ys else np.empty((0, 2))
            # scatters[name].set_offsets(coords)
        ax.set_title(f"Waypoints Path - Step {step}")
        return list(lines.values()) + list(scatters.values())

    ani = FuncAnimation(fig, update, frames=frames,
                        interval=interval, blit=True, repeat=False)

    if save_gif:
        ani.save(f"waypoints_animation.gif", writer="pillow", fps=5)
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", "-s", type=str, required=True)
    parser.add_argument("--time", "-t", type=int, default=100)
    args = parser.parse_args()

    route_history = {}
    base_dir = f"ViCo/meeting_challenge/output/{args.scene}/nav/curr_sim"
    for dir in os.listdir(base_dir):
        if 'Sentinel' in dir: continue
        subdir = os.path.join(base_dir, dir)
        if os.path.isdir(subdir):
            route_history[str(dir)] = json.load(open(os.path.join(subdir, "route_history.json"), "r"))

    animate_waypoints(route_history, args.scene, max_steps=args.time, interval=10, save_gif=True)
