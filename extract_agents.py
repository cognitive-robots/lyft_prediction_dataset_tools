#!/usr/bin/python3

import os
import argparse
import zarr
import json
import lz4.frame
import l5kit.data.filter
import numpy as np

import agent
import semantic_map

min_frame_count_threshold = 50

arg_parser = argparse.ArgumentParser(description="Extract and output agent data from Lyft Level 5 Prediction dataset")
arg_parser.add_argument("scene_dataset_dir_path")
arg_parser.add_argument("semantic_map_file_path")
arg_parser.add_argument("output_directory_path")
args = arg_parser.parse_args()

if not os.path.isdir(args.output_directory_path):
    raise ValueError(f"Output directory path {args.output_directory_path} is not a valid directory")

scene_dataset = zarr.open(args.scene_dataset_dir_path)

scene_data = scene_dataset["scenes"]
frame_data = scene_dataset["frames"]
agent_data = scene_dataset["agents"]

print("Scene dataset loaded")

loaded_semantic_map = semantic_map.SemanticMap.load(args.semantic_map_file_path)

for i in range(len(scene_data)):
    print(f"Processing scene {i+1} of {len(scene_data)}")

    frame_data_for_scene = frame_data[slice(*scene_data[i]["frame_index_interval"])]
    agent_data_for_scene = agent_data[frame_data_for_scene[0]["agent_index_interval"][0]:frame_data_for_scene[-1]["agent_index_interval"][1]]

    duration = (frame_data_for_scene[-1]["timestamp"] - frame_data_for_scene[0]["timestamp"]) / 1e9
    print(f"Scene is {duration} s long")

    scene_ego_agent = agent.Agent(scene_data[i]["frame_index_interval"][0], frame_data_for_scene, scene_data[i]["host"][-3:], agent_map=loaded_semantic_map)
    agents = [scene_ego_agent]

    scene_agent_track_ids = np.unique(agent_data_for_scene["track_id"])
    scene_non_ego_agents = list(map(lambda scene_agent_track_id : agent.Agent(scene_data[i]["frame_index_interval"][0], frame_data_for_scene, scene_agent_track_id, False, agent_data_for_scene, agent_map=loaded_semantic_map), scene_agent_track_ids))
    scene_non_ego_agents = [non_ego_agent for non_ego_agent in scene_non_ego_agents if len(non_ego_agent.states) >= min_frame_count_threshold]
    scene_non_ego_agents = [non_ego_agent for non_ego_agent in scene_non_ego_agents if non_ego_agent.class_label in l5kit.data.filter.PERCEPTION_LABELS_TO_KEEP]
    agents += scene_non_ego_agents

    with open(f"{args.output_directory_path}/scene-{i}.json.lz4", "wb") as output_file:
        json_str = json.dumps(agents, default=lambda obj : obj.toJSON())
        print("Converted agent data into JSON string")
        json_bytes = bytes(json_str, "utf-8")
        print("Converted JSON string to JSON bytes")
        lz4_json_bytes = lz4.frame.compress(json_bytes)
        print("Performed LZ4 compression on JSON bytes")
        output_file.write(lz4_json_bytes)
        print("Wrote LZ4 compressed JSON bytes to file")
