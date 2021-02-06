
import numpy as np
import bisect
import functools
import l5kit.rasterization
import l5kit.data.labels

import fluent

class Agent:

    def __init__(self, base_frame_index: int, frame_data: np.ndarray, agent_data: np.ndarray = None, track_id: int = -1):
        self.states = []

        if agent_data is None or track_id < 0:
            self.ego = True
            self.track_id = -1

            for current_frame_data_index in range(len(frame_data)):
                state = {}

                state["agent_index"] = -1
                state["frame_index"] = current_frame_data_index + base_frame_index
                state["timestamp"] = frame_data[current_frame_data_index]["timestamp"]
                state["translation"] = frame_data["ego_translation"]
                state["rotation"] = frame_data["ego_rotation"]
                state["extent"] = np.array([l5kit.rasterization.EGO_EXTENT_LENGTH, l5kit.rasterization.EGO_EXTENT_WIDTH, l5kit.rasterization.EGO_EXTENT_HEIGHT])
                state["class_probabilities"] = np.zeros(len(l5kit.data.labels.PERCEPTION_LABELS))
                state["class_probabilities"][l5kit.data.labels.PERCEPTION_LABELS.index("PERCEPTION_LABEL_CAR")] = 1.0
                state["class_label"] = "PERCEPTION_LABEL_CAR"

                self.states.append(state)

            if len(self.states) > 1:
                for i in range(len(self.states)):
                    if i == 0:
                        j = i + 1
                        k = i
                    elif i == len(self.states) - 1:
                        j = i
                        k = i - 1
                    else:
                        j = i + 1
                        k = i - 1
                    translation_diff = self.states[j]["translation"] - self.states[k]["translation"]
                    time_diff = (self.states[j]["timestamp"] - self.states[k]["timestamp"]) / 1e9

                    if time_diff == 0.0:
                        raise Exception("Timestamp difference of zero between frames {} and {}".format(self.states[k]["frame_index"], self.states[j]["frame_index"]))

                    self.states[i]["linear_velocity"] = translation_diff / time_diff

        else:
            self.ego = False
            self.track_id = track_id

            relevant_agent_data_indices = np.nonzero(agent_data["track_id"] == track_id)[0]

            base_agent_index = frame_data["agent_index_interval"][0, 0]
            cumulative_frames_agent_indices = frame_data["agent_index_interval"][:, 1]
            bisect_cumulative_frames_agent_indices = functools.partial(bisect.bisect_right, cumulative_frames_agent_indices)
            relevant_frame_data_indices = np.array(list(map(bisect_cumulative_frames_agent_indices, relevant_agent_data_indices + base_agent_index)))

            assert len(relevant_agent_data_indices) == len(relevant_frame_data_indices)

            for (current_agent_data_index, current_frame_data_index) in zip(relevant_agent_data_indices, relevant_frame_data_indices):
                state = {}

                state["agent_index"] = current_agent_data_index + base_agent_index
                state["frame_index"] = current_frame_data_index + base_frame_index
                state["timestamp"] = frame_data[current_frame_data_index]["timestamp"]
                state["translation"] = np.array([*agent_data[current_agent_data_index]["centroid"], 0])
                yaw = agent_data[current_agent_data_index]["yaw"]
                state["rotation"] = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
                state["extent"] = agent_data[current_agent_data_index]["extent"]
                state["class_probabilities"] = agent_data[current_agent_data_index]["label_probabilities"]
                state["class_label"] = l5kit.data.labels.PERCEPTION_LABELS[np.argmax(state["class_probabilities"])]

                self.states.append(state)

            for i in range(1, len(self.states) - 1):
                translation_diff = self.states[i + 1]["translation"] - self.states[i - 1]["translation"]
                time_diff = (self.states[i + 1]["timestamp"] - self.states[i - 1]["timestamp"]) / 1e9

                if time_diff == 0.0:
                    raise Exception("Timestamp difference of zero between frames {} and {}".format(self.states[k]["frame_index"], self.states[j]["frame_index"]))

                self.states[i]["linear_velocity"] = translation_diff / time_diff

            for i in range(2, len(self.states) - 2):
                linear_velocity_diff = self.states[i + 1]["linear_velocity"] - self.states[i - 1]["linear_velocity"]
                time_diff = (self.states[i + 1]["timestamp"] - self.states[i - 1]["timestamp"]) / 1e9

                if time_diff == 0.0:
                    raise Exception("Timestamp difference of zero between frames {} and {}".format(self.states[k]["frame_index"], self.states[j]["frame_index"]))

                self.states[i]["linear_acceleration"] = linear_velocity_diff / time_diff

            if len(self.states) > 4:
                self.states[0]["linear_acceleration"] = self.states[2]["linear_acceleration"]
                self.states[1]["linear_acceleration"] = self.states[2]["linear_acceleration"]
                self.states[-1]["linear_acceleration"] = self.states[-3]["linear_acceleration"]
                self.states[-2]["linear_acceleration"] = self.states[-3]["linear_acceleration"]
            elif len(self.states) == 4:
                linear_velocity_diff = self.states[2]["linear_velocity"] - self.states[1]["linear_velocity"]
                time_diff = (self.states[2]["timestamp"] - self.states[1]["timestamp"]) / 1e9
                linear_acceleration = linear_velocity_diff / time_diff
                self.states[0]["linear_acceleration"] = linear_acceleration
                self.states[1]["linear_acceleration"] = linear_acceleration
                self.states[2]["linear_acceleration"] = linear_acceleration
                self.states[3]["linear_acceleration"] = linear_acceleration
            else:
                for i in range(len(self.states)):
                    self.states[i]["linear_acceleration"] = np.zeros(3)

            if len(self.states) > 3:
                self.states[0]["linear_velocity"] = self.states[2]["linear_velocity"] - self.states[1]["linear_acceleration"] * ((self.states[2]["timestamp"] - self.states[0]["timestamp"]) / 1e9)
                self.states[-1]["linear_velocity"] = self.states[-3]["linear_velocity"] + self.states[-2]["linear_acceleration"] * ((self.states[-1]["timestamp"] - self.states[-3]["timestamp"]) / 1e9)
            elif len(self.states) == 3:
                self.states[0]["linear_velocity"] = self.states[1]["linear_velocity"]
                self.states[2]["linear_velocity"] = self.states[1]["linear_velocity"]
            elif len(self.states) == 2:
                translation_diff = self.states[1]["translation"] - self.states[0]["translation"]
                time_diff = (self.states[1]["timestamp"] - self.states[0]["timestamp"]) / 1e9
                linear_velocity = translation_diff / time_diff
                self.states[0]["linear_velocity"] = linear_velocity
                self.states[1]["linear_velocity"] = linear_velocity
            else:
                for i in range(len(self.states)):
                    self.states[i]["linear_velocity"] = np.zeros(3)

    def get_movement_fluent_changes(fluent_stability_window_threshold: float = 5.0, linear_velocity_threshold: float = 0.01, linear_acceleration_threshold: float = 0.01):
        fluent_changes = []
        for i in range(len(self.states) - 1):
            if np.linalg.norm(self.states[i]["linear_acceleration"]) < linear_acceleration_threshold:
                if np.linalg.norm(self.states[i]["linear_velocity"]) < linear_velocity_threshold:
                    current_fluent = fluent.MovementFluent.STATIONARY
                else:
                    current_fluent = fluent.MovementFluent.MOVING_CONSTANT
            else:
                if np.dot(self.states[i]["linear_velocity"], self.states[i]["linear_acceleration"]) < 0.0:
                    current_fluent = fluent.MovementFluent.MOVING_DECELERATING
                else:
                    current_fluent = fluent.MovementFluent.MOVING_ACCELERATING

            if len(fluent_changes) == 0:
                fluent_changes.append((self.states[i]["timestamp"] / 1e9, fluent.MovementFluent.UNKNOWN, current_fluent))

            if np.linalg.norm(self.states[i + 1]["linear_acceleration"]) < linear_acceleration_threshold:
                if np.linalg.norm(self.states[i + 1]["linear_velocity"]) < linear_velocity_threshold:
                    new_fluent = fluent.MovementFluent.STATIONARY
                else:
                    new_fluent = fluent.MovementFluent.MOVING_CONSTANT
            else:
                if np.dot(self.states[i + 1]["linear_velocity"], self.states[i + 1]["linear_acceleration"]) < 0.0:
                    new_fluent = fluent.MovementFluent.MOVING_DECELERATING
                else:
                    new_fluent = fluent.MovementFluent.MOVING_ACCELERATING

            if new_fluent != current_fluent:
                fluent_changes.append((self.states[i + 1]["timestamp"] / 1e9, current_fluent, new_fluent))

        # Add change to unknown fluent at the end

        return fluent_changes
