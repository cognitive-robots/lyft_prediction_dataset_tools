#!/usr/bin/python3

import os
import argparse
import json
import csv
import lz4.frame
import numpy as np

minimum_time_window_threshold = 10000
interpolation_count = 0

arg_parser = argparse.ArgumentParser(description="Converts a scene to a two agent follower scenario upon which causal discovery can be performed")
arg_parser.add_argument("scene_file_path")
arg_parser.add_argument("output_file_path")
arg_parser.add_argument("follower_agent_id", type=int)
arg_parser.add_argument("--all-kinematic-variables", action="store_true")
arg_parser.add_argument("--interagent-distance-variables", action="store_true")
arg_parser.add_argument("--independent-agent-ids", type=int, nargs="*", default=[])
args = arg_parser.parse_args()

if not os.path.isfile(args.scene_file_path):
    raise ValueError(f"Scene file path {args.scene_file_path} is not a valid file")

output_file_path_dir, _ = os.path.split(args.output_file_path)
if not os.path.isdir(output_file_path_dir):
    raise ValueError(f"Output file path parent directory {output_file_path_dir} is not a valid directory")

print(f"Opening file at file path: {args.scene_file_path}")
with open(args.scene_file_path, "rb") as input_file:
    lz4_json_bytes = input_file.read()
    print("Read LZ4 compressed JSON bytes from file")
    json_bytes = lz4.frame.decompress(lz4_json_bytes)
    print("Performed decompression on LZ4 compressed JSON bytes")
    json_str = json_bytes.decode("utf-8")
    print("Converted JSON bytes to JSON string")
    json_data = json.loads(json_str)
    print("Converted JSON string into Python processable structures")

    relevant_agents = {}
    latest_first_timestamp = None
    earliest_last_timestamp = None

    for agent in json_data:
        if agent["ego"] or agent["id"] == args.follower_agent_id or agent["id"] in args.independent_agent_ids:
            if agent["ego"]:
                relevant_agents["c1"] = agent
            elif agent["id"] == args.follower_agent_id:
                relevant_agents["c0"] = agent
            else:
                index = args.independent_agent_ids.index(agent["id"])
                relevant_agents[f"i{index}"] = agent

            if len(agent["states"]) < 1:
                agent_id = agent["id"]
                if agent["ego"]:
                    raise Exception(f"Ego agent with id {agent_id} has no states for the input scene file")
                else:
                    raise Exception(f"Non-ego agent with id {agent_id} has no states for the input scene file")

            if latest_first_timestamp is None or agent["states"][0]["timestamp"] > latest_first_timestamp:
                latest_first_timestamp = agent["states"][0]["timestamp"]

            if earliest_last_timestamp is None or agent["states"][-1]["timestamp"] < earliest_last_timestamp:
                earliest_last_timestamp = agent["states"][-1]["timestamp"]

    time_window = earliest_last_timestamp - latest_first_timestamp
    if time_window < minimum_time_window_threshold:
        raise Exception(f"Time window is {time_window} ms, which is below the minimum threshold of {minimum_time_window_threshold} ms")

    agent_ordering = []

    if not "c0" in relevant_agents:
        raise Exception("Leading agent not present in scene")
    else:
        agent_ordering.append("c0")

    if not "c1" in relevant_agents:
        raise Exception("Following agent not present in scene")
    else:
        agent_ordering.append("c1")

    for index in range(len(args.independent_agent_ids)):
        if not f"i{index}" in relevant_agents:
            raise Exception(f"Independent agent {index} not present in scene")
        else:
            agent_ordering.append(f"i{index}")


    agent_timeseries = {}
    agent_states = {}

    if args.all_kinematic_variables:
        for key in agent_ordering:
            agent_states[key] = []

            agent_timeseries[f"{key}.p"] = []
            agent_timeseries[f"{key}.v"] = []
            agent_timeseries[f"{key}.a"] = []

            states = relevant_agents[key]["states"]

            previous_position = None
            previous_distance_travelled = None
            previous_rotation_corrected_velocity = None
            previous_rotation_corrected_acceleration = None
            for state in states:
                if not (state["timestamp"] < latest_first_timestamp or state["timestamp"] > earliest_last_timestamp):
                    agent_states[key].append(state)

                    rotation = state["rotation"]
                    position = np.array(state["position"])
                    velocity = np.array(state["linear_velocity"])
                    acceleration = np.array(state["linear_acceleration"])

                    if previous_position is not None:
                        distance_travelled = previous_distance_travelled + np.linalg.norm(position - previous_position)
                    else:
                        distance_travelled = 0.0

                    if previous_distance_travelled is not None:
                        for i in range(1, interpolation_count + 1):
                            alpha = i / (interpolation_count + 1)
                            interpolated_distance_travelled = alpha * distance_travelled + (1 - alpha) * previous_distance_travelled
                            agent_timeseries[f"{key}.p"].append(interpolated_distance_travelled)

                    agent_timeseries[f"{key}.p"].append(distance_travelled)
                    previous_position = position
                    previous_distance_travelled = distance_travelled

                    rotation_corrected_velocity = np.cos(rotation) * velocity[0] + np.sin(rotation) * velocity[1]

                    if previous_rotation_corrected_velocity is not None:
                        for i in range(1, interpolation_count + 1):
                            alpha = i / (interpolation_count + 1)
                            interpolated_rotation_corrected_velocity = alpha * rotation_corrected_velocity + (1 - alpha) * previous_rotation_corrected_velocity
                            agent_timeseries[f"{key}.v"].append(1e3 * interpolated_rotation_corrected_velocity)

                    agent_timeseries[f"{key}.v"].append(1e3 * rotation_corrected_velocity)
                    previous_rotation_corrected_velocity = rotation_corrected_velocity

                    rotation_corrected_acceleration = np.cos(rotation) * acceleration[0] + np.sin(rotation) * acceleration[1]

                    if previous_rotation_corrected_acceleration is not None:
                        for i in range(1, interpolation_count + 1):
                            alpha = i / (interpolation_count + 1)
                            interpolated_rotation_corrected_acceleration = alpha * rotation_corrected_acceleration + (1 - alpha) * previous_rotation_corrected_acceleration
                            agent_timeseries[f"{key}.a"].append(1e6 * interpolated_rotation_corrected_acceleration)

                    agent_timeseries[f"{key}.a"].append(1e6 * rotation_corrected_acceleration)
                    previous_rotation_corrected_acceleration = rotation_corrected_acceleration
    else:
        for key in agent_ordering:
            agent_states[key] = []

            agent_timeseries[f"{key}.a"] = []

            states = relevant_agents[key]["states"]

            previous_rotation_corrected_acceleration = None
            for state in states:
                if not (state["timestamp"] < latest_first_timestamp or state["timestamp"] > earliest_last_timestamp):
                    agent_states[key].append(state)

                    rotation = state["rotation"]
                    acceleration = np.array(state["linear_acceleration"])
                    rotation_corrected_acceleration = np.cos(rotation) * acceleration[0] + np.sin(rotation) * acceleration[1]

                    if previous_rotation_corrected_acceleration is not None:
                        for i in range(1, interpolation_count + 1):
                            alpha = i / (interpolation_count + 1)
                            interpolated_rotation_corrected_acceleration = alpha * rotation_corrected_acceleration + (1 - alpha) * previous_rotation_corrected_acceleration
                            agent_timeseries[key].append(1e6 * interpolated_rotation_corrected_acceleration)

                    agent_timeseries[f"{key}.a"].append(1e6 * rotation_corrected_acceleration)
                    previous_rotation_corrected_acceleration = rotation_corrected_acceleration



    if args.interagent_distance_variables:
        for key in agent_ordering:
            states = agent_states[key]

            for other_key in agent_ordering:
                if other_key == key:
                    continue

                other_states = agent_states[other_key]

                if agent_timeseries.get(f"{key}-{other_key}.d") is None and agent_timeseries.get(f"{other_key}-{key}.d") is None:
                    if len(states) != len(other_states):
                        raise Exception(f"State count inconsistent across agents, {key} = {len(states)} vs {other_key} = {len(other_states)}")

                    agent_timeseries[f"{key}-{other_key}.d"] = []
                    previous_distance = None
                    for i in range(len(states)):
                        if states[i]["timestamp"] == other_states[i]["timestamp"]:
                            position_diff = np.array(states[i]["position"]) - np.array(other_states[i]["position"])
                            distance = np.linalg.norm(position_diff)

                            if previous_distance is not None:
                                for i in range(1, interpolation_count + 1):
                                    alpha = i / (interpolation_count + 1)
                                    interpolated_distance = alpha * distance + (1 - alpha) * previous_distance
                                    agent_timeseries[f"{key}-{other_key}.d"].append(interpolated_distance)

                            agent_timeseries[f"{key}-{other_key}.d"].append(distance)
                            previous_distance = distance

    ego_agent_timeseries_len = len(agent_timeseries["c1.a"])

    for key in agent_timeseries.keys():
        agent_timeseries_len = len(agent_timeseries[key])
        if agent_timeseries_len != ego_agent_timeseries_len:
            raise Exception(f"Timeseries length inconsistent across agents, c1.a = {ego_agent_timeseries_len} vs {key} = {agent_timeseries_len}")

    rows = []
    for i in range(ego_agent_timeseries_len):
        row = {}
        for key in agent_timeseries.keys():
            row[key] = agent_timeseries[key][i]
        rows.append(row)

    with open(args.output_file_path, 'w') as output_file:
        csv_writer = csv.DictWriter(output_file, fieldnames=agent_timeseries.keys())
        csv_writer.writeheader()
        for row in rows:
            csv_writer.writerow(row)
