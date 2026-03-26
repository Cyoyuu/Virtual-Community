import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image

def visualize_map_with_routes(
    img,
    new_route=None
):
    """
    danger_points: list of (x, y)
    original_route: ordered list of (x, y)
    new_route: ordered list of (x, y)
    """

    # -------------------------------------------------------
    # 2. Plot base image
    # -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img, origin='upper')
    # ax.set_title("Map Visualization")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # -------------------------------------------------------
    # 5. Draw new route (yellow)
    # -------------------------------------------------------
    # if new_route:
    #     xs, ys = zip(*new_route)
    #     ax.plot(xs, ys, color='orange', linewidth=1, label='New Route')
    #     ax.scatter(xs, ys, color='orange', s=7)

    # -------------------------------------------------------
    # 6. Legend entries for map semantics
    # -------------------------------------------------------
    legend_elements = []

    # White = open
    legend_elements.append(plt.Line2D([0], [0], marker='s', markersize=10,
                                      markerfacecolor='white', markeredgecolor='black',
                                      linestyle='None', label='Open Space (white)'))

    # Black = obstacle
    legend_elements.append(plt.Line2D([0], [0], marker='s', markersize=10,
                                      markerfacecolor='black', markeredgecolor='black',
                                      linestyle='None', label='Obstacle (black)'))

    # Red = danger
    legend_elements.append(plt.Line2D([0], [0], marker='o', markersize=10,
                                      markerfacecolor='red', linestyle='None',
                                      label='Danger (red)'))

    # Blue = original route
    legend_elements.append(plt.Line2D([0], [0], color='blue', linewidth=2, label='Original Route'))

    # Yellow = new route
    # legend_elements.append(plt.Line2D([0], [0], color='orange', linewidth=2, label='New Route'))

    ax.legend(handles=legend_elements, loc='upper right')

    plt.axis('off')
    plt.savefig("../icons/schematic_map_new.png", bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()


if __name__=="__main__":
    print(1)
    global_image = Image.open("../icons/Picture1.png").convert("RGB")
    print(2)
    visualize_map_with_routes(global_image, new_route=[[630, 300], [662, 406], [500, 400], [481, 415]])