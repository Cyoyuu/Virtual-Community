import json
import matplotlib.pyplot as plt
import argparse
import os

def plot_waypoints(route_history, steps=100):
    # 3. Create the plot
    plt.figure(figsize=(8, 8))
    plt.xlim(-400, 400)
    plt.ylim(-400, 400)
    colors = ['blue', 'red', 'yellow', 'cyan', 'green']
    x_coords={}
    y_coords={}
    for idx, name in enumerate(route_history):
        # 2. Extract IDs and locations
        ids = []
        x_coords[name] = []
        y_coords[name] = []
        for wp in route_history[name][f"{steps}"]:
            x_coords[name].append(wp[0])
            y_coords[name].append(wp[1])
        # Plot the path (line connecting waypoints)
        plt.plot(x_coords[name], y_coords[name], color=colors[idx], linewidth=2, label=f"Path - {name}", zorder=1)
    
        # Plot the individual waypoints
        plt.scatter(x_coords[name], y_coords[name], color='red', s=60, zorder=2, label=f"Waypoints - {name}")
        
        # Annotate each waypoint with its ID
        for i, (x, y) in enumerate(zip(x_coords[name], y_coords[name])):
            plt.annotate(str(i), (x, y), textcoords="offset points", xytext=(5,5),
                        ha='left', fontsize=9, color='darkblue')

    # Customize the graph
    plt.title("Waypoints Path", fontsize=14)
    plt.xlabel("X Coordinate", fontsize=12)
    plt.ylabel("Y Coordinate", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.axis('equal')  # Equal scaling for x and y axes
    plt.tight_layout()

    # Show the plot
    plt.savefig(f"step_{steps}_path.png")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output", "-o", type=str)
    args=parser.parse_args()
    route_history=dict()
    # Call the function
    for dir in os.listdir(args.output):
        if os.path.isdir(os.path.join(args.output, dir)):
            route_history[str(dir)]=json.load(open(os.path.join(args.output, dir, "route_history.json"), "r"))
    plot_waypoints(route_history, 100)
    