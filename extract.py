#!/usr/bin/python3

import os
import argparse
import zarr
import functools
import json
import numpy as np

import agent
import semantic_map

arg_parser = argparse.ArgumentParser(description="Extract data from Lyft Level 5 Prediction dataset")
arg_parser.add_argument("scene_dataset_dir_path")
arg_parser.add_argument("semantic_map_file_path")
arg_parser.add_argument("metadata_file_path")
args = arg_parser.parse_args()

if not os.path.isfile(args.metadata_file_path):
    raise ValueError("Metadata file path is not a valid file")

with open(args.metadata_file_path, "r") as metadata_file:
    metadata = json.load(metadata_file)
    world_to_ecef = np.array(metadata["world_to_ecef"])
    ecef_to_world = np.linalg.inv(world_to_ecef)

print("Metadata loaded")

if not os.path.isfile(args.semantic_map_file_path):
    raise ValueError("Semantic map file path is not a valid file")

loaded_semantic_map = semantic_map.SemanticMap(args.semantic_map_file_path, ecef_to_world)

print("Semantic map loaded")

if not os.path.isdir(args.scene_dataset_dir_path):
    raise ValueError("Scene dataset directory path is not a valid directory")

scene_dataset = zarr.open(args.scene_dataset_dir_path)

scene_data = scene_dataset["scenes"]
frame_data = scene_dataset["frames"]
agent_data = scene_dataset["agents"]
tl_face_data = scene_dataset["traffic_light_faces"]

print("Scene dataset loaded")

frame_data_by_scene = list(map(lambda frame_index_interval : frame_data[slice(*frame_index_interval)], scene_data["frame_index_interval"]))
agent_data_by_scene = list(map(lambda frame_data_for_scene: agent_data[frame_data_for_scene[0]["agent_index_interval"][0]:frame_data_for_scene[-1]["agent_index_interval"][1]], frame_data_by_scene))
tl_face_data_by_scene = list(map(lambda frame_data_for_scene: tl_face_data[frame_data_for_scene[0]["traffic_light_faces_index_interval"][0]:frame_data_for_scene[-1]["traffic_light_faces_index_interval"][1]], frame_data_by_scene))

agents = []

for i in range(len(scene_data)):
    scene_ego_agent = agent.Agent(scene_data[i]["frame_index_interval"][0], frame_data_by_scene[i])
    agents.append(scene_ego_agent)
    scene_agent_track_ids = np.unique(agent_data_by_scene[i]["track_id"])
    scene_non_ego_agents = list(map(lambda scene_agent_track_id : agent.Agent(scene_data[i]["frame_index_interval"][0], frame_data_by_scene[i], agent_data_by_scene[i], scene_agent_track_id), scene_agent_track_ids))
    agents.append(scene_non_ego_agents)

fluent_changes = []
for agent in agents:
    fluent_changes += agent.get_movement_fluent_changes()

print("Found {} movement fluent changes".format(len(fluent_changes)))
