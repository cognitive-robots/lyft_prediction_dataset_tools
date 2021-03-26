#!/usr/bin/python3

import os
import sys
import pickle
import argparse
import zarr
import json
import numpy as np

import semantic_map

from l5kit.data.proto import road_network_pb2
sys.modules["road_network_pb2"] = road_network_pb2

arg_parser = argparse.ArgumentParser(description="Extract and output map data from Lyft Level 5 Prediction dataset")
arg_parser.add_argument("scene_dataset_dir_path")
arg_parser.add_argument("semantic_map_file_path")
arg_parser.add_argument("metadata_file_path")
arg_parser.add_argument("output_file_path")
args = arg_parser.parse_args()

if not os.path.isfile(args.metadata_file_path):
    raise ValueError("Metadata file path is not a valid file")

with open(args.metadata_file_path, "r") as metadata_file:
    metadata = json.load(metadata_file)
    world_to_ecef = np.array(metadata["world_to_ecef"])
    ecef_to_world = np.linalg.inv(world_to_ecef)

print("Metadata loaded")

loaded_semantic_map = semantic_map.SemanticMap(args.semantic_map_file_path, args.scene_dataset_dir_path, ecef_to_world)

loaded_semantic_map.save(args.output_file_path)
