
import numpy as np
import bisect
import functools
import l5kit.rasterization
import l5kit.data.labels

import fluent
import semantic_map

class Agent:

    def __init__(self, base_frame_index: int, frame_data: np.ndarray, track_id: int, ego: bool = True, agent_data: np.ndarray = None, agent_map: semantic_map.SemanticMap = None, kinematic_timeseries_window: int = 15):
        self.states = []
        self.ego = ego
        self.id = int(track_id)

        if ego:
            self.bounding_box = [float(l5kit.rasterization.EGO_EXTENT_LENGTH), float(l5kit.rasterization.EGO_EXTENT_WIDTH)]
            self.class_label = "PERCEPTION_LABEL_CAR"

            for current_frame_data_index in range(len(frame_data)):
                state = {}

                #state["agent_index"] = -1
                #state["frame_index"] = current_frame_data_index + base_frame_index
                #state["timestamp"] = frame_data[current_frame_data_index]["timestamp"]
                #state["position"] = frame_data["ego_translation"]
                #state["rotation"] = frame_data["ego_rotation"]
                #state["extent"] = np.array([l5kit.rasterization.EGO_EXTENT_LENGTH, l5kit.rasterization.EGO_EXTENT_WIDTH, l5kit.rasterization.EGO_EXTENT_HEIGHT])
                #state["class_probabilities"] = np.zeros(len(l5kit.data.labels.PERCEPTION_LABELS))
                #state["class_probabilities"][l5kit.data.labels.PERCEPTION_LABELS.index("PERCEPTION_LABEL_CAR")] = 1.0
                #state["class_label"] = "PERCEPTION_LABEL_CAR"

                state["timestamp"] = int(frame_data[current_frame_data_index]["timestamp"] / 1e6)
                state["position"] = frame_data[current_frame_data_index]["ego_translation"][:2]
                state["rotation"] = np.arctan2(frame_data[current_frame_data_index]["ego_rotation"][1,0], frame_data[current_frame_data_index]["ego_rotation"][1,1])

                self.states.append(state)

        else:
            relevant_agent_data_indices = np.nonzero(agent_data["track_id"] == self.id)[0]
            base_agent_index = frame_data["agent_index_interval"][0, 0]
            cumulative_frames_agent_indices = frame_data["agent_index_interval"][:, 1]
            bisect_cumulative_frames_agent_indices = functools.partial(bisect.bisect_right, cumulative_frames_agent_indices)
            relevant_frame_data_indices = np.array(list(map(bisect_cumulative_frames_agent_indices, relevant_agent_data_indices + base_agent_index)))

            self.bounding_box = [float(np.max(agent_data["extent"][relevant_agent_data_indices,0])), float(np.max(agent_data["extent"][relevant_agent_data_indices,1]))]
            mean_class_confidence = np.mean(agent_data["label_probabilities"][relevant_agent_data_indices], axis=0)
            self.class_label = l5kit.data.labels.PERCEPTION_LABELS[np.argmax(mean_class_confidence)]

            for (current_agent_data_index, current_frame_data_index) in zip(relevant_agent_data_indices, relevant_frame_data_indices):
                state = {}

                #state["agent_index"] = current_agent_data_index + base_agent_index
                #state["frame_index"] = current_frame_data_index + base_frame_index
                #state["timestamp"] = frame_data[current_frame_data_index]["timestamp"]
                #state["position"] = np.array([*agent_data[current_agent_data_index]["centroid"], 0])
                #yaw = agent_data[current_agent_data_index]["yaw"]
                #state["rotation"] = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
                #state["extent"] = agent_data[current_agent_data_index]["extent"]
                #state["class_probabilities"] = agent_data[current_agent_data_index]["label_probabilities"]
                #state["class_label"] = l5kit.data.labels.PERCEPTION_LABELS[np.argmax(state["class_probabilities"])]

                state["timestamp"] = int(frame_data[current_frame_data_index]["timestamp"] / 1e6)
                state["position"] = agent_data[current_agent_data_index]["centroid"]
                state["rotation"] = agent_data[current_agent_data_index]["yaw"]

                self.states.append(state)



        if len(self.states) == 0:
            return

        kinematic_timeseries_window_wing = int((kinematic_timeseries_window - 1) / 2)



        for i in range(1, len(self.states) - 1):
            translation_diff = self.states[i + 1]["position"] - self.states[i - 1]["position"]
            rotation_diff_1 = self.states[i + 1]["rotation"] - self.states[i - 1]["rotation"]
            rotation_diff_2 = 2.0 * np.pi - (self.states[i + 1]["rotation"] - self.states[i - 1]["rotation"])
            rotation_diff = rotation_diff_1 if abs(rotation_diff_1) <= abs(rotation_diff_2) else rotation_diff_2
            time_diff = self.states[i + 1]["timestamp"] - self.states[i - 1]["timestamp"]

            if time_diff == 0.0:
                raise Exception("Timestamp difference of zero between frames {} and {}".format(i - 1, i + 1))

            self.states[i]["linear_velocity"] = translation_diff / time_diff
            self.states[i]["angular_velocity"] = rotation_diff / time_diff

        if len(self.states) < 2:
            self.states[0]["linear_velocity"] = np.zeros(2)
            self.states[0]["angular_velocity"] = 0.0
        elif len(self.states) == 2:
            translation_diff = self.states[1]["position"] - self.states[0]["position"]
            rotation_diff_1 = self.states[1]["rotation"] - self.states[0]["rotation"]
            rotation_diff_2 = 2.0 * np.pi - (self.states[1]["rotation"] - self.states[0]["rotation"])
            rotation_diff = rotation_diff_1 if abs(rotation_diff_1) <= abs(rotation_diff_2) else rotation_diff_2
            time_diff = self.states[1]["timestamp"] - self.states[0]["timestamp"]

            if time_diff == 0.0:
                raise Exception("Timestamp difference of zero between frames 0 and 1")

            self.states[0]["linear_velocity"] = translation_diff / time_diff
            self.states[0]["angular_velocity"] = rotation_diff / time_diff
            self.states[1]["linear_velocity"] = np.copy(self.states[0]["linear_velocity"])
            self.states[1]["angular_velocity"] = np.copy(self.states[0]["angular_velocity"])
        else:
            self.states[0]["linear_velocity"] = np.copy(self.states[1]["linear_velocity"])
            self.states[len(self.states) - 1]["linear_velocity"] = np.copy(self.states[len(self.states) - 2]["linear_velocity"])
            self.states[0]["angular_velocity"] = np.copy(self.states[1]["angular_velocity"])
            self.states[len(self.states) - 1]["angular_velocity"] = np.copy(self.states[len(self.states) - 2]["angular_velocity"])

        for i in range(len(self.states)):
            self.states[i]["linear_velocity_moving_mean"] = np.copy(self.states[i]["linear_velocity"])
            #temp = self.states[i]["linear_velocity"]
            #print(temp)
            #temp = self.states[i]["linear_velocity_moving_mean"]
            #print(f"{i}: {temp}")
            self.states[i]["angular_velocity_moving_mean"] = np.copy(self.states[i]["angular_velocity"])
            for j in range(1, kinematic_timeseries_window_wing + 1):
                k = max(i - j, 0)
                self.states[i]["linear_velocity_moving_mean"] += self.states[k]["linear_velocity"]
                #temp = self.states[k]["linear_velocity"]
                #print(temp)
                #temp = self.states[i]["linear_velocity_moving_mean"]
                #print(f"{i}-{j} = {k}: {temp}")
                self.states[i]["angular_velocity_moving_mean"] += self.states[k]["angular_velocity"]
            for j in range(1, kinematic_timeseries_window_wing + 1):
                k = min(i + j, len(self.states) - 1)
                self.states[i]["linear_velocity_moving_mean"] += self.states[k]["linear_velocity"]
                #temp = self.states[k]["linear_velocity"]
                #print(temp)
                #temp = self.states[i]["linear_velocity_moving_mean"]
                #print(f"{i}+{j} = {k}: {temp}")
                self.states[i]["angular_velocity_moving_mean"] += self.states[k]["angular_velocity"]
            #temp = self.states[i]["linear_velocity_moving_mean"]
            #print(f"sum: {temp}")
            self.states[i]["linear_velocity_moving_mean"] /= 2 * kinematic_timeseries_window_wing + 1
            #temp = self.states[i]["linear_velocity_moving_mean"]
            #print(f"mean: {temp}")
            self.states[i]["angular_velocity_moving_mean"] /= 2 * kinematic_timeseries_window_wing + 1

        for i in range(len(self.states)):
            self.states[i]["linear_velocity"] = self.states[i]["linear_velocity_moving_mean"]
            del self.states[i]["linear_velocity_moving_mean"]
            self.states[i]["angular_velocity"] = self.states[i]["angular_velocity_moving_mean"]
            del self.states[i]["angular_velocity_moving_mean"]



        for i in range(1, len(self.states) - 1):
            linear_velocity_diff = self.states[i + 1]["linear_velocity"] - self.states[i - 1]["linear_velocity"]
            angular_velocity_diff = self.states[i + 1]["angular_velocity"] - self.states[i - 1]["angular_velocity"]
            time_diff = self.states[i + 1]["timestamp"] - self.states[i - 1]["timestamp"]

            self.states[i]["linear_acceleration"] = linear_velocity_diff / time_diff
            self.states[i]["angular_acceleration"] = angular_velocity_diff / time_diff

        if len(self.states) < 3:
            for i in range(len(self.states)):
                self.states[i]["linear_acceleration"] = np.zeros(2)
                self.states[i]["angular_acceleration"] = 0.0
        else:
            self.states[0]["linear_acceleration"] = np.copy(self.states[1]["linear_acceleration"])
            self.states[len(self.states) - 1]["linear_acceleration"] = np.copy(self.states[len(self.states) - 2]["linear_acceleration"])
            self.states[0]["angular_acceleration"] = np.copy(self.states[1]["angular_acceleration"])
            self.states[len(self.states) - 1]["angular_acceleration"] = np.copy(self.states[len(self.states) - 2]["angular_acceleration"])

        for i in range(len(self.states)):
            self.states[i]["linear_acceleration_moving_mean"] = np.copy(self.states[i]["linear_acceleration"])
            self.states[i]["angular_acceleration_moving_mean"] = np.copy(self.states[i]["angular_acceleration"])
            for j in range(1, kinematic_timeseries_window_wing + 1):
                k = max(i - j, 0)
                self.states[i]["linear_acceleration_moving_mean"] += self.states[k]["linear_acceleration"]
                self.states[i]["angular_acceleration_moving_mean"] += self.states[k]["angular_acceleration"]
            for j in range(1, kinematic_timeseries_window_wing + 1):
                k = min(i + j, len(self.states) - 1)
                self.states[i]["linear_acceleration_moving_mean"] += self.states[k]["linear_acceleration"]
                self.states[i]["angular_acceleration_moving_mean"] += self.states[k]["angular_acceleration"]
            self.states[i]["linear_acceleration_moving_mean"] /= 2 * kinematic_timeseries_window_wing + 1
            self.states[i]["angular_acceleration_moving_mean"] /= 2 * kinematic_timeseries_window_wing + 1

        for i in range(len(self.states)):
            self.states[i]["linear_acceleration"] = self.states[i]["linear_acceleration_moving_mean"]
            del self.states[i]["linear_acceleration_moving_mean"]
            self.states[i]["angular_acceleration"] = self.states[i]["angular_acceleration_moving_mean"]
            del self.states[i]["angular_acceleration_moving_mean"]

        #if map is not None:
        #    for i in range(len(self.states)):
        #        position = self.states[i]["position"]
        #        x = position[0]
        #        y = position[1]
        #        lane = agent_map.get_encapsulating_lane((x, y))
        #        if lane is not None:
        #            self.states[i]["lane"] = lane.get_global_id()
        #        else:
        #            self.states[i]["lane"] = ""

        for i in range(len(self.states)):
            self.states[i]["position"] = list(map(float, self.states[i]["position"]))
            self.states[i]["rotation"] = float(self.states[i]["rotation"])
            self.states[i]["linear_velocity"] = list(map(float, self.states[i]["linear_velocity"]))
            self.states[i]["angular_velocity"] = float(self.states[i]["angular_velocity"])
            self.states[i]["linear_acceleration"] = list(map(float, self.states[i]["linear_acceleration"]))
            self.states[i]["angular_acceleration"] = float(self.states[i]["angular_acceleration"])

    def toJSON(self):
        return self.__dict__
