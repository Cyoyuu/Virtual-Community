import json
import os
import pdb
import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
from PIL import Image
import argparse
import sys
import time

from pathlib import Path

current_directory = os.getcwd()
sys.path.insert(0, current_directory)


def lat_lon_to_xy(lat, lon, ref_lat, ref_lon):
    earth_radius = 6378137  # in meters
    meters_per_degree_lat = 111139  # Approximate meters per degree latitude
    
    # Calculate meters per degree longitude based on reference latitude
    meters_per_degree_lon = 111139 * math.cos(math.radians(ref_lat))
    
    # Convert lat/lon to x/y
    x = (lon - ref_lon) * meters_per_degree_lon
    y = (lat - ref_lat) * meters_per_degree_lat
    
    return [x, y]

def draw_line_segment(start, end, width, color='white'):
    # Calculate the direction vector from start to end
    direction = np.array(end) - np.array(start)
    length = np.linalg.norm(direction)
    
    # Normalize the direction vector
    if length == 0:
        return  # Avoid division by zero if start and end are the same
    direction /= length
    
    # Calculate perpendicular vector
    perp_direction = np.array([-direction[1], direction[0]])  # Rotate 90 degrees
    
    # Calculate offset for width
    offset = (width / 2) * perp_direction
    
    # Define the four corners of the line segment rectangle
    corner1 = start + offset
    corner2 = start - offset
    corner3 = end - offset
    corner4 = end + offset
    
    # Create a polygon (rectangle) representing the line segment with width
    line_segment = np.array([corner1, corner2, corner3, corner4])
    
    # Plotting
    plt.fill(line_segment[:, 0], line_segment[:, 1], color=color)  # Draw filled rectangle for width
    plt.plot([start[0], end[0]], [start[1], end[1]], color='black', linewidth=1)  # Draw main line

def draw_roads(roads, nodes, ref_lat, ref_lon):
    color_dict={'primary': 'red', 'secondary': 'orangered', 'tertiary': 'orange', 'residential': 'gold', 'cycleway':'green', 'pedestrian': 'blue', 'footway': 'deepskyblue', 'service': 'peru', 'unclassified': 'm', 'steps': 'blue', 'elevator':'violet', 'living_street':'pink', 'construction': 'orchid'}
    for road in roads:
        # print(road)
        highway = road['highway']
        color = 'grey'
        if highway in color_dict:
            color = color_dict[highway]
        # color = {'yes': 'red', 'no': 'blue'}[road['oneway']]
        start_x, start_y = road['start']['x'],road['start']['y']
        end_x, end_y = road['end']['x'],road['end']['y']
        # print(road['width'])
        # assert not isinstance(road['width'],str)
        draw_line_segment([start_x, start_y], [end_x, end_y] ,road['width'], color)
    # for node in nodes.values():
    #     plt.scatter(node['x'], node['y'], color='lime', s=2)

# python tools/fetch_osm_roads.py -s newyork --ref_lat 40.748998486718186 --ref_lon -73.9882893780644 --radius 400
# python tools/fetch_osm_roads.py -s detroit --ref_lat 42.33165461030516 --ref_lon -83.0480662316049 --radius 400
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", "-s", type=str, required=True)
    # parser.add_argument("--radius", type=float, required=True)
    args = parser.parse_args()
    if os.path.exists(f"ViCo/assets/scenes/{args.scene}"):
        print("Scene exists")
        Path(f"ViCo/assets/scenes/{args.scene}/road_data").mkdir(parents=True, exist_ok=True)
        with open(f"ViCo/assets/scenes/{args.scene}/raw/center.txt", "r") as f:
            ref_lat, ref_lon=f.readline().split()
            args.ref_lat, args.ref_lon=float(ref_lat), float(ref_lon)
            print(f"retrieved coordinates from raw file. lat: {args.ref_lat}, lon: {args.ref_lon}")
    else:
        print(f"Scene not exist: ViCo/assets/scenes/{args.scene}")
        exit()

    # Set up the plot area
    plt.figure(figsize=(8, 8))
    aerial_view=mpimg.imread(f"ViCo/assets/scenes/{args.scene}/global.png")
    plt.imshow(aerial_view, extent=[-512, 512, -512, 512])# left, right, bottom, top
    plt.xlim(-400, 400)
    plt.ylim(-400, 400)
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.grid()

    roads, nodes = pickle.load(open(f"ViCo/assets/scenes/{args.scene}/road_data/roads.pkl", 'rb'))
    draw_roads(roads, nodes, args.ref_lat, args.ref_lon)

    color_dict = {'primary': 'red', 'secondary': 'orangered', 'tertiary': 'orange', 'residential': 'gold', 
              'cycleway':'green', 'pedestrian': 'blue', 'footway': 'deepskyblue', 'service': 'peru', 
              'unclassified': 'm', 'steps': 'blue', 'elevator':'violet', 'living_street':'pink', 'construction': 'orchid'}

    legend_handles = [mpatches.Patch(color=color, label=road_type) for road_type, color in color_dict.items()]

    plt.legend(handles=legend_handles, title="Road Types", loc='upper left', fontsize='small', title_fontsize='medium')

    # Show the plot
    # plt.title(f'Roads with Width - {args.scene}')
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    plt.show()