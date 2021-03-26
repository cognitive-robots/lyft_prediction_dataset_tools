#!/usr/bin/python3

import os
import argparse
import zarr
import numpy as np
import cv2 as cv
import l5kit.configs
import l5kit.data
import l5kit.dataset
import l5kit.rasterization

arg_parser = argparse.ArgumentParser(description="Outputs a visualisation of a given frame from a scene")
arg_parser.add_argument("dataset_directory_path")
arg_parser.add_argument("config_file_path")
arg_parser.add_argument("scene_number", type=int)
arg_parser.add_argument("frame_number", type=int)
arg_parser.add_argument("output_file_path")
arg_parser.add_argument("--frame_number_offsets", nargs="*", type=int, default=[0])
args = arg_parser.parse_args()

os.environ["L5KIT_DATA_FOLDER"] = args.dataset_directory_path
data_manager = l5kit.data.LocalDataManager()

config = l5kit.configs.load_config_data(args.config_file_path)
dataset_path = data_manager.require(config["val_data_loader"]["key"])
scene_dataset = l5kit.data.ChunkedDataset(dataset_path)
scene_dataset.open()

rasterizer = l5kit.rasterization.build_rasterizer(config, data_manager)
ego_dataset = l5kit.dataset.EgoDataset(config, scene_dataset, rasterizer)

for i in args.frame_number_offsets:
    if i == 0:
        output_file_path = args.output_file_path
    else:
        head, tail = os.path.splitext(args.output_file_path)
        output_file_path = head + "-offset_{}_frames".format(i) + tail

    frame_data = ego_dataset.get_frame(args.scene_number, args.frame_number + i)

    image = frame_data["image"].transpose(1, 2, 0)
    image = ego_dataset.rasterizer.to_rgb(image)

    cv.imwrite(output_file_path, image)
