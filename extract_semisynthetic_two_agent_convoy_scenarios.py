#!/usr/bin/python3

import os
import math
import argparse
import csv
import json
import lz4.frame
import numpy as np

cols = ["c0.a", "c1.a", "i0.a"]
meta_cols = ["output_file_path", "duration"]
positional_threshold = 5 # m
angular_threshold = 0.866 # 1, Defined as ~cos(30) degrees
temporal_threshold = 1000 # ms
framerate = 10 # Hz

def output_csv(output_file, cols, rows):
    csv_writer = csv.DictWriter(output_file, fieldnames=cols)
    csv_writer.writeheader()
    for row in rows:
        csv_writer.writerow(row)

def input_json_lz4(input_file):
    lz4_json_bytes = input_file.read()
    json_bytes = lz4.frame.decompress(lz4_json_bytes)
    json_str = json_bytes.decode("utf-8")
    json_data = json.loads(json_str)
    return json_data

arg_parser = argparse.ArgumentParser(description="Takes an agent scene directory and outputs semi-synthetic two agent convoy scenarios")
arg_parser.add_argument("input_dir_path")
arg_parser.add_argument("output_dir_path")
args = arg_parser.parse_args()

if not os.path.isdir(args.input_dir_path):
    raise Exception(f"Input directory path '{args.input_dir_path}' is not a valid directory")

if not os.path.isdir(args.output_dir_path):
    raise Exception(f"Input directory path '{args.output_dir_path}' is not a valid directory")

temporal_separation = 1000 / framerate

i = 0
j = 0
rows = []
meta_rows = []
need_independent = False

while True:
    last_timestamp = None
    ego_last_position = None
    other_convoy_agent = None
    other_convoy_agent_last_position = None
    ego_lead = None
    current_independent_row = 0

    output_file_path = os.path.join(args.output_dir_path, f"scene-{i}.csv")

    while True:
        input_file_path = os.path.join(args.input_dir_path, f"scene-{j}.json.lz4")
        if not os.path.isfile(input_file_path):
            break
        else:
            with open(input_file_path, 'rb') as input_file:
                print(f"Processing input file '{input_file_path}'")
                scene = input_json_lz4(input_file)

                ego_agent = next(filter(lambda agent: agent["ego"], scene))
                non_ego_agents = list(filter(lambda agent: not agent["ego"], scene))

                if len(ego_agent["states"]) == 0:
                    print("Failed to continue due to no states")
                    break

                ego_timestamp = ego_agent["states"][0]["timestamp"]
                timestamp_difference = ego_timestamp - last_timestamp if last_timestamp is not None else temporal_separation
                if last_timestamp is not None and timestamp_difference > temporal_threshold:
                    print(f"Failed to continue due to mismatched inter-scene ego timestamps: {ego_timestamp} - {last_timestamp} = {timestamp_difference} > {temporal_threshold}")
                    break
                last_timestamp = ego_agent["states"][-1]["timestamp"]

                ego_position = np.array(ego_agent["states"][0]["position"])
                if ego_last_position is not None and np.linalg.norm(ego_position - ego_last_position) > positional_threshold:
                    print(f"Failed to continue due to mismatched inter-scene ego positions: np.linalg.norm({ego_position}) - {ego_last_position}) = {np.linalg.norm(ego_position - ego_last_position)} > {positional_threshold}")
                    break
                ego_last_position = np.array(ego_agent["states"][-1]["position"])

                if need_independent:
                    if current_independent_row > 0:
                        steps = round(timestamp_difference / temporal_separation)
                        previous_acceleration = rows[current_independent_row - 1]["i0.a"]
                        next_acceleration = np.linalg.norm(np.array(ego_agent["states"][0]["linear_acceleration"]) * 1e6)
                        for k in range(1, steps):
                            alpha = k / steps
                            if current_independent_row >= len(rows):
                                break
                            else:
                                rows[current_independent_row]["i0.a"] = alpha * next_acceleration + (1 - alpha) * previous_acceleration
                                current_independent_row += 1


                    for k in range(len(ego_agent["states"])):
                        if current_independent_row >= len(rows):
                            break
                        else:
                            rows[current_independent_row]["i0.a"] = np.linalg.norm(np.array(ego_agent["states"][k]["linear_acceleration"]) * 1e6)
                            current_independent_row += 1
                    if current_independent_row >= len(rows):
                        break
                else:
                    if other_convoy_agent_last_position is not None:
                        other_convoy_agent_candidates = filter(lambda non_ego_agent: non_ego_agent["states"][0]["timestamp"] == ego_agent["states"][0]["timestamp"] and len(non_ego_agent["states"]) == len(ego_agent["states"]), non_ego_agents)
                        other_convoy_agent_candidates = list(filter(lambda other_convoy_agent_candidate: np.linalg.norm(np.array(other_convoy_agent_candidate["states"][0]["position"]) - other_convoy_agent_last_position) <= positional_threshold, other_convoy_agent_candidates))

                        if len(other_convoy_agent_candidates) != 1:
                            print("Failed to continue due to losing other convoy agent between scenes")
                            break

                        other_convoy_agent = other_convoy_agent_candidates[0]

                        relative_position = np.array(other_convoy_agent["states"][0]["position"]) - np.array(ego_agent["states"][0]["position"])
                        relative_position_unit = relative_position / np.linalg.norm(relative_position)

                        ego_rotation_unit = np.array((np.cos(ego_agent["states"][0]["rotation"]), np.sin(ego_agent["states"][0]["rotation"])))

                        if abs(np.dot(relative_position_unit, ego_rotation_unit)) < angular_threshold:
                            print("Failed to continue due to other convoy agent not lying within the front or rear cone of the ego agent")
                            break

                        if (np.dot(relative_position_unit, ego_rotation_unit) < 0) != ego_lead:
                            print("Failed to continue due to other convoy agent not maintaining convoy order between scenes")
                            break
                    else:
                        other_convoy_agent_candidates = sorted(non_ego_agents, key=lambda non_ego_agent: np.linalg.norm(np.array(non_ego_agent["states"][0]["position"]) - ego_agent["states"][0]["position"]))

                        for other_convoy_agent_candidate in other_convoy_agent_candidates:
                            if other_convoy_agent_candidate["states"][0]["timestamp"] != ego_agent["states"][0]["timestamp"] or len(other_convoy_agent_candidate["states"]) != len(ego_agent["states"]):
                                continue

                            relative_position = np.array(other_convoy_agent_candidate["states"][0]["position"]) - np.array(ego_agent["states"][0]["position"])
                            relative_position_unit = relative_position / np.linalg.norm(relative_position)

                            ego_rotation_unit = np.array((np.cos(ego_agent["states"][0]["rotation"]), np.sin(ego_agent["states"][0]["rotation"])))

                            if abs(np.dot(relative_position_unit, ego_rotation_unit)) >= angular_threshold:
                                other_convoy_agent = other_convoy_agent_candidate
                                break

                        if other_convoy_agent is None:
                            print("Failed to continue due to inability to find other convoy agent")
                            break
                    other_convoy_agent_last_position = other_convoy_agent["states"][-1]["position"]
                    ego_lead = (np.dot(relative_position_unit, ego_rotation_unit) < 0)

                    if len(rows) > 0:
                        steps = round(timestamp_difference / temporal_separation)
                        if ego_lead:
                            previous_ego_acceleration = rows[-1]["c0.a"]
                            next_ego_acceleration = np.linalg.norm(np.array(ego_agent["states"][0]["linear_acceleration"]) * 1e6)
                            previous_other_convoy_agent_acceleration = rows[-1]["c1.a"]
                            next_other_convoy_agent_acceleration = np.linalg.norm(np.array(other_convoy_agent["states"][0]["linear_acceleration"]) * 1e6)
                            for k in range(1, steps):
                                alpha = k / steps
                                row = {
                                "c0.a": alpha * next_ego_acceleration + (1 - alpha) * previous_ego_acceleration,
                                "c1.a": alpha * next_other_convoy_agent_acceleration + (1 - alpha) * previous_other_convoy_agent_acceleration
                                }
                                rows.append(row)
                        else:
                            previous_ego_acceleration = rows[-1]["c1.a"]
                            next_ego_acceleration = np.linalg.norm(np.array(ego_agent["states"][0]["linear_acceleration"]) * 1e6)
                            previous_other_convoy_agent_acceleration = rows[-1]["c0.a"]
                            next_other_convoy_agent_acceleration = np.linalg.norm(np.array(other_convoy_agent["states"][0]["linear_acceleration"]) * 1e6)
                            for k in range(1, steps):
                                alpha = k / steps
                                row = {
                                "c0.a": alpha * next_other_convoy_agent_acceleration + (1 - alpha) * previous_other_convoy_agent_acceleration,
                                "c1.a": alpha * next_ego_acceleration + (1 - alpha) * previous_ego_acceleration
                                }
                                rows.append(row)

                    for k in range(len(ego_agent["states"])):
                        if ego_lead:
                            row = {
                            "c0.a": np.linalg.norm(np.array(ego_agent["states"][k]["linear_acceleration"]) * 1e6),
                            "c1.a": np.linalg.norm(np.array(other_convoy_agent["states"][k]["linear_acceleration"]) * 1e6)
                            }
                        else:
                            row = {
                            "c0.a": np.linalg.norm(np.array(other_convoy_agent["states"][k]["linear_acceleration"]) * 1e6),
                            "c1.a": np.linalg.norm(np.array(ego_agent["states"][k]["linear_acceleration"]) * 1e6)
                            }
                        rows.append(row)


        j += 1

    if len(rows) > 0:
        if need_independent == False:
            need_independent = True
        else:
            need_independent = False
            if current_independent_row < len(rows):
                rows = rows[:current_independent_row]
            with open(output_file_path, 'w') as output_file:
                print(f"Processing output file '{output_file_path}'")
                output_csv(output_file, cols, rows)
            meta_rows.append({ "output_file_path": output_file_path, "duration": ((len(rows) - 1) * temporal_separation) / 1e3 })
            rows = []
            i += 1
    else:
        j += 1

    if not os.path.isfile(input_file_path):
        break

output_file_path = os.path.join(args.output_dir_path, "meta.csv")
with open(output_file_path, 'w') as output_file:
    print(f"Processing output file '{output_file_path}'")
    output_csv(output_file, meta_cols, meta_rows)
