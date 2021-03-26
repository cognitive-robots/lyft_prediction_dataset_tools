
from __future__ import annotations

import os
import pickle
import copy
import math
import itertools
import shapely.geometry
import zarr
import json
import lzo
import numpy as np
import l5kit.data.proto.road_network_pb2 as rnpb
import matplotlib.pyplot as plt

import fluent

import pymap3d as pm

TL_FACE_TYPE_TL_FLUENT_DICT = {
"signal_flashing_yellow": fluent.TrafficLightFluent.YELLOW,
"signal_flashing_red": fluent.TrafficLightFluent.RED,
"signal_red_face": fluent.TrafficLightFluent.RED,
"signal_yellow_face": fluent.TrafficLightFluent.YELLOW,
"signal_green_face": fluent.TrafficLightFluent.GREEN,
"signal_left_arrow_red_face": fluent.TrafficLightFluent.RED,
"signal_left_arrow_yellow_face": fluent.TrafficLightFluent.YELLOW,
"signal_left_arrow_green_face": fluent.TrafficLightFluent.GREEN,
"signal_right_arrow_red_face": fluent.TrafficLightFluent.RED,
"signal_right_arrow_yellow_face": fluent.TrafficLightFluent.YELLOW,
"signal_right_arrow_green_face": fluent.TrafficLightFluent.GREEN,
"signal_upper_left_arrow_red_face": fluent.TrafficLightFluent.RED,
"signal_upper_left_arrow_yellow_face": fluent.TrafficLightFluent.YELLOW,
"signal_upper_left_arrow_green_face": fluent.TrafficLightFluent.GREEN,
"signal_upper_right_arrow_red_face": fluent.TrafficLightFluent.RED,
"signal_upper_right_arrow_yellow_face": fluent.TrafficLightFluent.YELLOW,
"signal_upper_right_arrow_green_face": fluent.TrafficLightFluent.GREEN,
"signal_red_u_turn": fluent.TrafficLightFluent.RED,
"signal_yellow_u_turn": fluent.TrafficLightFluent.YELLOW,
"signal_green_u_turn": fluent.TrafficLightFluent.GREEN
}


def decode_global_id(global_id: rnpb.GlobalId, encoding: str = "utf-8") -> str:
    return global_id.id.decode(encoding)

def decompress_geoframe_origin(geoframe: rnpb.GeoFrame) -> np.ndarray:
    return np.array([geoframe.origin.lat_e7 / 1e7, geoframe.origin.lng_e7 / 1e7, geoframe.origin.altitude_cm / 100])

def convert_rel_to_abs(rel_coord_array: np.ndarray) -> np.ndarray:
    return np.cumsum(rel_coord_array, axis=-1)

def convert_local_enu_to_world(local_enu_coord_array: np.ndarray, local_enu_frame_geodetic_datum: np.ndarray, ecef_to_world: np.ndarray) -> np.ndarray:
    ecef_coord_array = np.stack(pm.enu2ecef(local_enu_coord_array[0], local_enu_coord_array[1], local_enu_coord_array[2], local_enu_frame_geodetic_datum[0], local_enu_frame_geodetic_datum[1], local_enu_frame_geodetic_datum[2]))

    homogeneous_ecef_coord_array = np.concatenate((ecef_coord_array, np.ones((1, ecef_coord_array.shape[-1]))))
    homogeneous_world_coord_array = ecef_to_world @ homogeneous_ecef_coord_array
    world_coord_array = homogeneous_world_coord_array[:3]

    return world_coord_array

def get_coord_min_projection(coord_array: np.ndarray, coord_to_project: np.ndarray) -> np.ndarray:
    assert len(coord_array.shape) == 2
    assert len(coord_to_project.shape) == 1
    assert coord_array.shape[0] == coord_to_project.shape[0]

    coord_projections = coord_array.T - coord_to_project
    coord_projection_norms = np.linalg.norm(coord_projections, axis=-1)
    min_coord_projection_norm_index = np.argmin(coord_projection_norms)
    min_coord_projection = coord_projections[min_coord_projection_norm_index]

    if min_coord_projection_norm_index == 0:
        if min_coord_projection_norm_index == coord_array.shape[-1] - 1:
            min_projection = min_coord_projection
            projection_indices = (min_coord_projection_norm_index, None)
        else:
            next_coord_projection = coord_projections[min_coord_projection_norm_index + 1]
            min_next_diff = next_coord_projection - min_coord_projection
            min_next_diff_unit = min_next_diff / np.linalg.norm(min_next_diff)
            length_at_next_projection = np.dot(min_next_diff_unit, -min_coord_projection)

            if length_at_next_projection >= 0.0:
                min_projection = (min_next_diff_unit * length_at_next_projection) + min_coord_projection
                projection_indices = (min_coord_projection_norm_index, min_coord_projection_norm_index + 1)
            else:
                min_projection = min_coord_projection
                projection_indices = (min_coord_projection_norm_index, None)
    elif min_coord_projection_norm_index == coord_array.shape[-1] - 1:
        previous_coord_projection = coord_projections[min_coord_projection_norm_index - 1]
        min_previous_diff = previous_coord_projection - min_coord_projection
        min_previous_diff_unit = min_previous_diff / np.linalg.norm(min_previous_diff)
        length_at_previous_projection = np.dot(min_previous_diff_unit, -min_coord_projection)

        if length_at_previous_projection >= 0.0:
            min_projection = (min_previous_diff_unit * length_at_previous_projection) + min_coord_projection
            projection_indices = (min_coord_projection_norm_index, min_coord_projection_norm_index - 1)
        else:
            min_projection = min_coord_projection
            projection_indices = (min_coord_projection_norm_index, None)
    else:
        next_coord_projection = coord_projections[min_coord_projection_norm_index + 1]
        min_next_diff = next_coord_projection - min_coord_projection
        min_next_diff_unit = min_next_diff / np.linalg.norm(min_next_diff)
        length_at_next_projection = np.dot(min_next_diff_unit, -min_coord_projection)

        previous_coord_projection = coord_projections[min_coord_projection_norm_index - 1]
        min_previous_diff = previous_coord_projection - min_coord_projection
        min_previous_diff_unit = min_previous_diff / np.linalg.norm(min_previous_diff)
        length_at_previous_projection = np.dot(min_previous_diff_unit, -min_coord_projection)

        if length_at_next_projection >= 0.0 or length_at_previous_projection >= 0.0:
            if length_at_next_projection >= length_at_previous_projection:
                min_projection = (min_next_diff_unit * length_at_next_projection) + min_coord_projection
                projection_indices = (min_coord_projection_norm_index, min_coord_projection_norm_index + 1)
            else:
                min_projection = (min_previous_diff_unit * length_at_previous_projection) + min_coord_projection
                projection_indices = (min_coord_projection_norm_index, min_coord_projection_norm_index - 1)
        else:
            min_projection = min_coord_projection
            projection_indices = (min_coord_projection_norm_index, None)

    return min_projection, projection_indices

def line_intersects_coord_array(coord_array: np.ndarray, line_origin_coord: np.ndarray, line_direction_vector: np.ndarray) -> bool:
    coord_projections = coord_array.T - coord_to_project
    line_direction_vector_normal = np.array([[0, -1], [1, 0]]) * line_direction_vector
    valid_coords = np.dot(line_direction_vector, coord_projections) > 0.0
    return (np.any(valid_coords & (np.dot(line_direction_vector_normal, coord_projections) >= 0.0))
        and np.any(valid_coords & (np.dot(line_direction_vector_normal, coord_projections) <= 0.0)))

class LaneTreeInterface:

    def get_aerial_centroid(self) -> (float, float):
        raise Exception("This is an interface class not meant to be called")

    def get_coord_count(self) -> int:
        raise Exception("This is an interface class not meant to be called")

    def get_aerial_bb(self) -> (float, float, float, float):
        raise Exception("This is an interface class not meant to be called")

    def get_encapsulating_lane(self, aerial_coord: (float, float)) -> Lane:
        raise Exception("This is an interface class not meant to be called")

class Lane:

    def __init__(self, global_id: str, semantic_map: SemanticMap, lane_data: rnpb.Lane = None, ecef_to_world: np.ndarray = None, json_data = None):
        if (lane_data is not None) and (ecef_to_world is not None):
            local_enu_frame_geodetic_datum = decompress_geoframe_origin(lane_data.geo_frame)

            self.left_boundary_coord_array = convert_local_enu_to_world(
                convert_rel_to_abs(0.01 * np.stack((
                    np.array(lane_data.left_boundary.vertex_deltas_x_cm),
                    np.array(lane_data.left_boundary.vertex_deltas_y_cm),
                    np.array(lane_data.left_boundary.vertex_deltas_z_cm)))),
                local_enu_frame_geodetic_datum,
                ecef_to_world)
            self.left_boundary_coord_array = np.float64(self.left_boundary_coord_array)
            self.left_boundary_aerial_centroid = tuple(np.mean(self.left_boundary_coord_array[:2], axis=-1))
            self.left_boundary_aerial_bb = (np.min(self.left_boundary_coord_array[0]),
                                            np.min(self.left_boundary_coord_array[1]),
                                            np.max(self.left_boundary_coord_array[0]),
                                            np.max(self.left_boundary_coord_array[1]))

            self.right_boundary_coord_array = convert_local_enu_to_world(
                convert_rel_to_abs(0.01 * np.stack((
                    np.array(lane_data.right_boundary.vertex_deltas_x_cm),
                    np.array(lane_data.right_boundary.vertex_deltas_y_cm),
                    np.array(lane_data.right_boundary.vertex_deltas_z_cm)))),
                local_enu_frame_geodetic_datum,
                ecef_to_world)
            self.right_boundary_coord_array = np.float64(self.right_boundary_coord_array)
            self.right_boundary_aerial_centroid = tuple(np.mean(self.right_boundary_coord_array[:2], axis=-1))
            self.right_boundary_aerial_bb = (np.min(self.right_boundary_coord_array[0]),
                                            np.min(self.right_boundary_coord_array[1]),
                                            np.max(self.right_boundary_coord_array[0]),
                                            np.max(self.right_boundary_coord_array[1]))

            self.coord_count = self.left_boundary_coord_array.shape[-1] + self.right_boundary_coord_array.shape[-1]
            self.aerial_centroid = tuple((self.left_boundary_coord_array.shape[-1] * np.asarray(self.left_boundary_aerial_centroid) +
                self.right_boundary_coord_array.shape[-1] * np.asarray(self.right_boundary_aerial_centroid)) / self.coord_count)
            self.aerial_bb = (min(self.left_boundary_aerial_bb[0], self.right_boundary_aerial_bb[0]),
                min(self.left_boundary_aerial_bb[1], self.right_boundary_aerial_bb[1]),
                max(self.left_boundary_aerial_bb[2], self.right_boundary_aerial_bb[2]),
                max(self.left_boundary_aerial_bb[3], self.right_boundary_aerial_bb[3]))

            self.access_restriction = int(lane_data.access_restriction.type)
            self.adjacent_left_id = decode_global_id(lane_data.adjacent_lane_change_left)
            self.adjacent_right_id = decode_global_id(lane_data.adjacent_lane_change_right)
            self.ahead_ids = list(map(decode_global_id, lane_data.lanes_ahead))
            self.behind_ids = []
            self.traffic_control_ids = list(map(decode_global_id, lane_data.traffic_controls))
        elif json_data is not None:
            self.__dict__ = json_data
            for key in self.__dict__.keys():
                if key in [ "coord_array", "left_boundary_coord_array", "right_boundary_coord_array" ]:
                    self.__dict__[key] = np.asarray(self.__dict__[key]).T
        else:
            raise ValueError("Must either provide combination of protobuf lane data and a ecef to world frame transform or JSON data")

        self.global_id = global_id
        self.semantic_map = semantic_map

    def get_global_id(self) -> str:
        return self.global_id

    def get_semantic_map(self) -> SemanticMap:
        return self.semantic_map

    def get_left_boundary_coord_array(self) -> np.ndarray:
        return self.left_boundary_coord_array

    def get_right_boundary_coord_array(self) -> np.ndarray:
        return self.right_boundary_coord_array

    def get_aerial_centroid(self) -> (float, float):
        return self.aerial_centroid

    def get_coord_count(self) -> int:
        return self.coord_count

    def get_aerial_bb(self) -> (float, float, float, float):
        return self.aerial_bb

    def get_encapsulating_lane(self, aerial_coord: (float, float)) -> Lane:
        if (aerial_coord[0] >= self.aerial_bb[0] and aerial_coord[1] >= self.aerial_bb[1]
            and aerial_coord[0] <= self.aerial_bb[2] and aerial_coord[1] <= self.aerial_bb[3]):
            if np.linalg.norm(self.left_boundary_coord_array[:2,0] - self.right_boundary_coord_array[:2,0]) + \
                np.linalg.norm(self.left_boundary_coord_array[:2,-1] - self.right_boundary_coord_array[:2,-1]) <= \
                np.linalg.norm(self.left_boundary_coord_array[:2,0] - self.right_boundary_coord_array[:2,-1]) + \
                np.linalg.norm(self.left_boundary_coord_array[:2,-1] - self.right_boundary_coord_array[:2,0]):
                self.polygon = shapely.geometry.Polygon(np.concatenate((self.left_boundary_coord_array[:2].T, self.right_boundary_coord_array[:2,::-1].T)))
            else:
                self.polygon = shapely.geometry.Polygon(np.concatenate((self.left_boundary_coord_array[:2].T, self.right_boundary_coord_array[:2].T)))

            point = shapely.geometry.Point(aerial_coord)
            point_contained = self.polygon.intersects(point)

            # Uncomment to see lane polygons and the point the encapsulation check is being carried out on (always)
            #lane_polygon_plot_x, lane_polygon_plot_y = self.polygon.exterior.xy
            #plt.plot(self.left_boundary_coord_array[0], self.left_boundary_coord_array[1], 'r', linewidth=3)
            #plt.plot(self.right_boundary_coord_array[0], self.right_boundary_coord_array[1], 'g', linewidth=3)
            #plt.plot(lane_polygon_plot_x, lane_polygon_plot_y)
            #plt.plot(point.x, point.y, "ro")
            #plt.gca().set_aspect('equal', adjustable='box')
            #plt.show()

            if point_contained:
            	return self
            else:
                # Uncomment to see lane polygons and the point the encapsulation check is being carried out on (failure only)
                #lane_polygon_plot_x, lane_polygon_plot_y = self.polygon.exterior.xy
                #plt.plot(self.left_boundary_coord_array[0], self.left_boundary_coord_array[1], 'r', linewidth=3)
                #plt.plot(self.right_boundary_coord_array[0], self.right_boundary_coord_array[1], 'g', linewidth=3)
                #plt.plot(lane_polygon_plot_x, lane_polygon_plot_y)
                #plt.plot(point.x, point.y, "yo")
                #plt.gca().set_aspect('equal', adjustable='box')
                #plt.show()
                return None
        else:
            return None

    def get_access_restriction(self) -> rnpb.AccessRestriction.Type:
        return rnpb.AccessRestriction.Type(self.access_restriction)

    def get_left_adjacent_lane(self) -> Lane:
        return self.semantic_map.get_lane(self.adjacent_left_id)

    def get_right_adjacent_lane(self) -> Lane:
        return self.semantic_map.get_lane(self.adjacent_right_id)

    def get_lanes_ahead(self) -> [Lane]:
        return list(map(self.semantic_map.get_lane, self.ahead_ids))

    def get_lanes_behind(self) -> [Lane]:
        return list(map(self.semantic_map.get_lane, self.behind_ids))

    def append_behind_id(self, new_behind_id):
        self.behind_ids.append(new_behind_id)

    def get_traffic_lights(self) -> [TrafficLight]:
        return [traffic_light for traffic_light in list(map(self.semantic_map.get_traffic_light, self.traffic_control_ids)) if traffic_light is not None]

    def toJSON(self):
        property_dict = self.__dict__.copy()

        for key in [ "global_id", "semantic_map" ]:
            if property_dict.get(key) is not None:
                del property_dict[key]

        for key in [ "coord_array", "left_boundary_coord_array", "right_boundary_coord_array" ]:
            if property_dict.get(key) is not None:
                property_dict[key] = list(map(list, property_dict.get(key).T))

        return property_dict


class LaneProxy(LaneTreeInterface):

    def __init__(self, lane: Lane = None, global_id: str = None, semantic_map: SemanticMap = None):
        if lane is not None:
            self.global_id = lane.get_global_id()
            self.semantic_map = lane.get_semantic_map()
        elif (global_id is not None) and (semantic_map is not None):
            self.global_id = global_id
            self.semantic_map = semantic_map
        else:
            raise ValueError("Must either provide lane instance or a combination of global id and semantic map instance")

    def get_aerial_centroid(self) -> (float, float):
        return self.semantic_map.get_lane(self.global_id).get_aerial_centroid()

    def get_coord_count(self) -> int:
        return self.semantic_map.get_lane(self.global_id).get_coord_count()

    def get_aerial_bb(self) -> (float, float, float, float):
        return self.semantic_map.get_lane(self.global_id).get_aerial_bb()

    def get_encapsulating_lane(self, aerial_coord: (float, float)) -> Lane:
        return self.semantic_map.get_lane(self.global_id).get_encapsulating_lane(aerial_coord)

    def toJSON(self):
        return { "global_id": self.global_id }


class LaneTreeBinaryBranch(LaneTreeInterface):

    def __init__(self, left_branch: LaneTreeInterface, right_branch: LaneTreeInterface, json_data = None):
        if json_data is None:
            self.left_branch = left_branch
            self.right_branch = right_branch

            self.coord_count = self.left_branch.get_coord_count() + self.right_branch.get_coord_count()
            self.aerial_centroid = tuple((self.left_branch.get_coord_count() * np.asarray(self.left_branch.get_aerial_centroid()) + self.right_branch.get_coord_count() * np.asarray(self.right_branch.get_aerial_centroid())) / self.coord_count)
            left_aerial_bb = self.left_branch.get_aerial_bb()
            right_aerial_bb = self.right_branch.get_aerial_bb()
            self.aerial_bb = (min(left_aerial_bb[0], right_aerial_bb[0]),
                min(left_aerial_bb[1], right_aerial_bb[1]),
                max(left_aerial_bb[2], right_aerial_bb[2]),
                max(left_aerial_bb[3], right_aerial_bb[3]))
        else:
            self.__dict__ = json_data
            self.left_branch = left_branch
            self.right_branch = right_branch

    def get_aerial_centroid(self) -> (float, float):
        return self.aerial_centroid

    def get_coord_count(self) -> int:
        return self.coord_count

    def get_aerial_bb(self) -> (float, float, float, float):
        return self.aerial_bb

    def get_encapsulating_lane(self, aerial_coord: (float, float)) -> Lane:
        if (aerial_coord[0] >= self.aerial_bb[0] and aerial_coord[1] >= self.aerial_bb[1]
            and aerial_coord[0] <= self.aerial_bb[2] and aerial_coord[1] <= self.aerial_bb[3]):
            left_branch_result = self.left_branch.get_encapsulating_lane(aerial_coord)
            if left_branch_result is not None:
                return left_branch_result
            else:
                right_branch_result = self.right_branch.get_encapsulating_lane(aerial_coord)
                if right_branch_result is not None:
                    return right_branch_result
                else:
                    return None
        else:
            return None

    def toJSON(self):
        return self.__dict__


class LaneTreeKBranch(LaneTreeInterface):

    def __init__(self, branches: [LaneTreeInterface], json_data = None):
        if json_data is None:
            self.branches = branches

            self.coord_count = sum(map(lambda branch : branch.get_coord_count(), self.branches))
            self.aerial_centroid = tuple(sum(map(lambda branch : branch.get_coord_count() * np.asarray(branch.get_aerial_centroid()), self.branches)) / self.coord_count)
            aerial_bbs = np.array(list(map(lambda branch : branch.get_aerial_bb(), self.branches)))
            self.aerial_bb = (min(aerial_bbs[:,0]),
                min(aerial_bbs[:,1]),
                max(aerial_bbs[:,2]),
                max(aerial_bbs[:,3]))
        else:
            self.__dict__ = json_data
            self.branches = branches

    def get_aerial_centroid(self) -> (float, float):
        return self.aerial_centroid

    def get_coord_count(self) -> int:
        return self.coord_count

    def get_aerial_bb(self) -> (float, float, float, float):
        return self.aerial_bb

    def get_encapsulating_lane(self, aerial_coord: (float, float)) -> Lane:
        if (aerial_coord[0] >= self.aerial_bb[0] and aerial_coord[1] >= self.aerial_bb[1]
            and aerial_coord[0] <= self.aerial_bb[2] and aerial_coord[1] <= self.aerial_bb[3]):
            for branch in self.branches:
                branch_result = branch.get_encapsulating_lane(aerial_coord)
                if branch_result is not None:
                    return branch_result
        else:
            return None

    def toJSON(self):
        return self.__dict__


class TrafficLight:

    def __init__(self, global_id: str, semantic_map: SemanticMap, traffic_control_element_data: rnpb.TrafficControlElement = None, ecef_to_world: np.ndarray = None, json_data = None):
        if (traffic_control_element_data is not None) and (ecef_to_world is not None):
            assert traffic_control_element_data.geometry_type == rnpb.TrafficControlElement.GeometryType.UKNOWN

            local_enu_frame_geodetic_datum = decompress_geoframe_origin(traffic_control_element_data.geo_frame)
            self.bearing_degrees = traffic_control_element_data.geo_frame.bearing_degrees

            self.coord = tuple(convert_local_enu_to_world(
                np.zeros((3, 1)),
                local_enu_frame_geodetic_datum,
                ecef_to_world)[:,0])

            self.traffic_light_face_ids = list(map(decode_global_id, traffic_control_element_data.traffic_light.face_states))
        elif json_data is not None:
            self.__dict__ = json_data
        else:
            raise ValueError("Must either provide combination of protobuf traffic control data and a ecef to world frame transform or JSON data")

        self.global_id = global_id
        self.semantic_map = semantic_map

    def get_global_id(self) -> str:
        return self.global_id

    def get_coord(self) -> (float, float, float):
        return self.coord

    def get_bearing_degrees(self) -> float:
        return self.bearing_degrees

    def get_traffic_light_faces(self) -> [TrafficLightFace]:
        return list(map(self.semantic_map.get_traffic_light_face, self.traffic_light_face_ids))

    def get_active_traffic_light_face(self, timestamp: float) -> TrafficLightFace:
        if len(self.traffic_light_face_ids) == 0:
            return None
        traffic_light_face_fluents = map(lambda global_id : self.semantic_map.get_traffic_light_face_fluent(global_id, timestamp), self.traffic_light_face_ids)
        active_traffic_light_faces = [traffic_light_face_fluent_pair[0] for traffic_light_face_fluent_pair in zip(self.get_traffic_light_faces(), traffic_light_face_fluents) if traffic_light_face_fluent_pair[1] == fluent.TrafficLightFaceFluent.ACTIVE]
        if len(active_traffic_light_faces) == 0:
            return None
        else:
            return active_traffic_light_faces[0]

    def get_traffic_light_fluent_changes(self) -> [(float, fluent.TrafficLightFluent, fluent.TrafficLightFluent)]:
        if len(self.traffic_light_face_ids) == 0:
            return []
        traffic_light_timestamps = self.semantic_map.get_traffic_light_timestamps(self.global_id)
        active_traffic_light_faces = map(self.get_active_traffic_light_face, traffic_light_timestamps)
        active_traffic_light_face_types = map(lambda traffic_light_face : traffic_light_face.get_type() if traffic_light_face is not None else None, active_traffic_light_faces)
        traffic_light_fluents = map(lambda traffic_light_face_type : TL_FACE_TYPE_TL_FLUENT_DICT.get(traffic_light_face_type, fluent.TrafficLightFluent.UNKNOWN), active_traffic_light_face_types)
        traffic_light_frames = zip(traffic_light_timestamps, traffic_light_fluents)

        fluent_changes = []
        previous_fluent = fluent.TrafficLightFluent.UNKNOWN
        for current_frame in traffic_light_frames:
            if current_frame[1] != previous_fluent:
                fluent_changes.append((current_frame[0], previous_fluent, current_frame[1]))

            previous_fluent = current_frame[1]

        return fluent_changes

    def toJSON(self):
        property_dict = self.__dict__.copy()

        for key in [ "global_id", "semantic_map" ]:
            if property_dict.get(key) is not None:
                del property_dict[key]

        return property_dict


class TrafficLightFace:

    def __init__(self, global_id: str, semantic_map: SemanticMap, traffic_control_element_data: rnpb.TrafficControlElement = None, traffic_light_face_type: str = None, json_data = None):
        if (traffic_control_element_data is not None) and (traffic_light_face_type is not None):
            traffic_light_face_data = getattr(traffic_control_element_data, traffic_light_face_type)

            self.traffic_light_face_type = traffic_light_face_type
            self.observing_lane_ids = [decode_global_id(yield_rule.lane) for yield_rule in traffic_light_face_data.yield_rules_when_on]
            self.observing_lane_id_yield_to_lane_ids_dict = {decode_global_id(yield_rule.lane): list(map(decode_global_id, yield_rule.yield_to_lanes)) for yield_rule in traffic_light_face_data.yield_rules_when_on}
            self.observing_lane_id_yield_to_crosswalk_ids_dict = {decode_global_id(yield_rule.lane): list(map(decode_global_id, yield_rule.yield_to_crosswalks)) for yield_rule in traffic_light_face_data.yield_rules_when_on}
            self.no_right_turn = traffic_light_face_data.no_right_turn_on_red

            assert not self.no_right_turn or traffic_light_face_type.find("red") >= 0
        elif json_data is not None:
            self.__dict__ = json_data
        else:
            raise ValueError("Must either provide combination of protobuf traffic control data and a traffic light face type string or JSON data")

        self.global_id = global_id
        self.semantic_map = semantic_map

    def get_global_id(self) -> str:
        return self.global_id

    def get_type(self) -> str:
        return self.traffic_light_face_type

    def get_parent(self) -> TrafficLight:
        return self.semantic_map.get_parent_traffic_light(self.global_id)

    def get_observing_lanes(self) -> [Lane]:
        return map(self.semantic_map.get_lane, self.observing_lane_id)

    def get_yield_to_lanes_for_observing_lane(self, lane: Lane) -> [Lane]:
        return map(self.semantic_map.get_lane, self.observing_lane_id_yield_to_lane_ids_dict.get(lane.global_id))

    def get_traffic_light_face_fluent_changes(self) -> [(float, fluent.TrafficLightFaceFluent, fluent.TrafficLightFaceFluent)]:
        traffic_light_face_frames = self.semantic_map.get_traffic_light_face_frames(self.global_id)

        if traffic_light_face_frames is None:
            return []

        fluent_changes = []
        previous_fluent = fluent.TrafficLightFaceFluent.UNKNOWN
        for current_frame in traffic_light_face_frames:
            if current_frame[1] != previous_fluent:
                fluent_changes.append((current_frame[0], previous_fluent, current_frame[1]))

            previous_fluent = current_frame[1]

        return fluent_changes

    def toJSON(self):
        property_dict = self.__dict__.copy()

        for key in [ "global_id", "semantic_map" ]:
            if property_dict.get(key) is not None:
                del property_dict[key]

        return property_dict


class SemanticMap:

    def __init__(self, raw_map_pb_file_path: str = None, scene_zarr_dir_path: str = None, ecef_to_world: np.ndarray = None, json_data = None):
        if json_data is not None:
            self.id_traffic_light_face_frames_dict = json_data["id_traffic_light_face_frames_dict"]
            self.id_traffic_light_timestamps = json_data["id_traffic_light_timestamps"]
            self.id_traffic_light_face_id_traffic_light_dict = json_data["id_traffic_light_face_id_traffic_light_dict"]

            json_id_lane_dict = json_data["id_lane_dict"]
            self.id_lane_dict = {}
            for id in json_id_lane_dict.keys():
                self.id_lane_dict[id] = Lane(id, self, json_data=json_id_lane_dict[id])

            json_id_traffic_light_dict = json_data["id_traffic_light_dict"]
            self.id_traffic_light_dict = {}
            for id in json_id_traffic_light_dict.keys():
                self.id_traffic_light_dict[id] = TrafficLight(id, self, json_data=json_id_traffic_light_dict[id])

            json_id_traffic_light_face_dict = json_data["id_traffic_light_face_dict"]
            self.id_traffic_light_face_dict = {}
            for id in json_id_traffic_light_face_dict.keys():
                self.id_traffic_light_face_dict[id] = TrafficLightFace(id, self, json_data=json_id_traffic_light_face_dict[id])

            self.lane_tree = self.load_lane_tree(json_data["lane_tree"])

            print("Loaded semantic map JSON data")
        elif (raw_map_pb_file_path is not None) and (scene_zarr_dir_path is not None) and (ecef_to_world is not None):
            if not os.path.isfile(raw_map_pb_file_path):
                raise ValueError("Semantic map protobuf file path is not a valid file")

            if not os.path.isdir(scene_zarr_dir_path):
                raise ValueError("Scene dataset directory path is not a valid directory")

            with open(raw_map_pb_file_path, "rb") as map_pb_file:
                map_fragment = rnpb.MapFragment()
                map_fragment.ParseFromString(map_pb_file.read())

            scene_dataset = zarr.open(scene_zarr_dir_path)

            frame_data = scene_dataset["frames"]
            tl_face_data = scene_dataset["traffic_light_faces"]

            self.id_traffic_light_face_frames_dict = {}
            self.id_traffic_light_timestamps = {}
            self.id_traffic_light_face_id_traffic_light_dict = {}

            counter = 0
            frame_deltatime_sum = 0
            previous_timestamp = None
            for current_frame_data in frame_data:
                print("Processed {0:.2f}% of frames".format(100 * counter / frame_data.shape[0]), end="\r")
                for tl_face_index in range(current_frame_data["traffic_light_faces_index_interval"][0], current_frame_data["traffic_light_faces_index_interval"][1]):
                    current_tl_face_data = tl_face_data[tl_face_index]

                    if self.id_traffic_light_face_frames_dict.get(current_tl_face_data["face_id"]) is None:
                        self.id_traffic_light_face_frames_dict[current_tl_face_data["face_id"]] = []
                    self.id_traffic_light_face_frames_dict[current_tl_face_data["face_id"]].append((current_frame_data["timestamp"] / 1e9,
                        fluent.TrafficLightFaceFluent(int(fluent.TrafficLightFaceFluent.ACTIVE) - np.nonzero(current_tl_face_data["traffic_light_face_status"])[0][0])))

                    if self.id_traffic_light_timestamps.get(current_tl_face_data["traffic_light_id"]) is None:
                        self.id_traffic_light_timestamps[current_tl_face_data["traffic_light_id"]] = []
                    if len(self.id_traffic_light_timestamps[current_tl_face_data["traffic_light_id"]]) == 0 \
                        or self.id_traffic_light_timestamps[current_tl_face_data["traffic_light_id"]][-1] != current_frame_data["timestamp"] / 1e9:
                        self.id_traffic_light_timestamps[current_tl_face_data["traffic_light_id"]].append(current_frame_data["timestamp"] / 1e9)

                    self.id_traffic_light_face_id_traffic_light_dict[current_tl_face_data["face_id"]] = current_tl_face_data["traffic_light_id"]

                if previous_timestamp is not None:
                    frame_deltatime_sum += current_frame_data["timestamp"] / 1e9 - previous_timestamp

                previous_timestamp = current_frame_data["timestamp"] / 1e9

                counter += 1
            frame_deltatime_mean = frame_deltatime_sum / counter
            print("")
            print("Finished processing frames")

            self.id_lane_dict = {}
            self.id_traffic_light_dict = {}
            self.id_traffic_light_face_dict = {}

            counter = 0
            for element in map_fragment.elements:
                print("Processed {0:.2f}% of map elements".format(100 * counter / len(map_fragment.elements)), end="\r")

                global_id = decode_global_id(element.id)

                if bool(element.element.HasField("lane")):
                    lane = Lane(global_id, self, element.element.lane, ecef_to_world)
                    self.id_lane_dict[global_id] = lane

                if bool(element.element.HasField("traffic_control_element")):
                    traffic_control_element_type = element.element.traffic_control_element.WhichOneof("Type")

                    if traffic_control_element_type == "traffic_light":
                        traffic_light = TrafficLight(global_id, self, element.element.traffic_control_element, ecef_to_world)
                        self.id_traffic_light_dict[global_id] = traffic_light

                    if traffic_control_element_type in TL_FACE_TYPE_TL_FLUENT_DICT.keys():
                        traffic_light_face = TrafficLightFace(global_id, self, element.element.traffic_control_element, traffic_control_element_type)
                        self.id_traffic_light_face_dict[global_id] = traffic_light_face

                counter += 1
            print("")
            print("Finished processing map elements")

            counter = 0
            for lane in self.id_lane_dict.values():
                print("Post-processed {0:.2f}% of lanes".format(100 * counter / len(self.id_lane_dict.values())), end="\r")

                lane_global_id = lane.get_global_id()

                lanes_ahead = lane.get_lanes_ahead()
                for lane_ahead in lanes_ahead:
                    lane_ahead.append_behind_id(lane_global_id)

                counter += 1
            print("")
            print("Finished post-processing lanes")

            counter = 0
            for traffic_light_face in self.id_traffic_light_face_dict.values():
                print("Post-processed {0:.2f}% of traffic light faces".format(100 * counter / len(self.id_traffic_light_face_dict.values())), end="\r")

                traffic_light_face_frames = self.id_traffic_light_face_frames_dict.get(traffic_light_face.get_global_id())
                if traffic_light_face_frames is not None:
                    if traffic_light_face_frames[-1][1] != fluent.TrafficLightFaceFluent.UNKNOWN:
                        new_timestamp = traffic_light_face_frames[-1][0] + frame_deltatime_mean
                        self.id_traffic_light_face_frames_dict[traffic_light_face.get_global_id()].append(
                            (new_timestamp, fluent.TrafficLightFaceFluent.UNKNOWN))
                        if self.id_traffic_light_timestamps[self.id_traffic_light_face_id_traffic_light_dict[traffic_light_face.get_global_id()]][-1] != new_timestamp:
                            self.id_traffic_light_timestamps[self.id_traffic_light_face_id_traffic_light_dict[traffic_light_face.get_global_id()]].append(new_timestamp)

                counter += 1
            print("")
            print("Finished post-processing traffic light faces")

            lane_proxies = list(map(lambda lane : LaneProxy(lane=lane), self.id_lane_dict.values()))
            self.lane_tree = self.construct_lane_tree(lane_proxies)
            print("Finished constructing lane tree")
        else:
            raise ValueError("Must either provide raw map/scene data to process or JSON data from a saved semantic map")

        print("Finished constructing semantic map")

    def save(self, output_file_path: str) -> None:
        output_file_path_dir, _ = os.path.split(output_file_path)

        if not os.path.isdir(output_file_path_dir):
            raise ValueError("Output file path directory {} is not a valid directory".format(output_file_path_dir))

        with open(output_file_path, "wb") as output_file:
            json_str = json.dumps(self, default=lambda obj : obj.toJSON())
            print("Converted semantic map into JSON string")
            lzo_json_str = lzo.compress(json_str, 5)
            print("Performed LZO compression on JSON string")
            output_file.write(lzo_json_str)
            print("Wrote LZO compressed JSON string to file")

    @staticmethod
    def load(input_file_path: str) -> SemanticMap:
        input_file_path_dir, _ = os.path.split(input_file_path)

        if not os.path.isdir(input_file_path_dir):
            raise ValueError("Input file path directory {} is not a valid directory".format(input_file_path_dir))

        with open(input_file_path, "rb") as input_file:
            lzo_json_str = input_file.read()
            print("Read LZO compressed JSON string from file")
            json_str = lzo.decompress(lzo_json_str)
            print("Performed decompression on LZO compressed JSON string")
            json_data = json.loads(json_str)
            print("Converted JSON string into Python processable structures")
            return SemanticMap(json_data=json_data)

    def toJSON(self):
        return self.__dict__

    def load_lane_tree(self, json_data):
        if json_data.get("global_id") is not None:
            return LaneProxy(global_id=json_data["global_id"], semantic_map=self)
        elif (json_data.get("left_branch") is not None) and (json_data.get("right_branch") is not None):
            left_branch = self.load_lane_tree(json_data["left_branch"])
            right_branch = self.load_lane_tree(json_data["right_branch"])
            return LaneTreeBinaryBranch(left_branch, right_branch, json_data=json_data)
        elif json_data.get("branches") is not None:
            branches = list(map(self.load_lane_tree, json_data["branches"]))
            return LaneTreeKBranch(branches, json_data=json_data)

    def construct_lane_tree(self, lanes: [LaneProxy], dimension: int = 0, dimstop: int = -1) -> LaneTreeInterface:
        if dimension >= 2:
            raise ValueError("Dimension index out of bounds")

        if len(lanes) > 1:
            lane_aerial_centroids = np.asarray(list(map(lambda lane : lane.get_aerial_centroid(), lanes))).T
            dimension_median = np.median(lane_aerial_centroids[dimension])
            left_branch_lanes = list(itertools.compress(lanes, lane_aerial_centroids[dimension] < dimension_median))
            right_branch_lanes = list(itertools.compress(lanes, lane_aerial_centroids[dimension] >= dimension_median))
            next_dimension = (dimension + 1) % 2
            if len(left_branch_lanes) == 0 or len(right_branch_lanes) == 0:
            	if dimstop == -1:
            	    return self.construct_lane_tree(lanes, next_dimension, dimension)
            	elif dimension == dimstop:
            	    return LaneTreeKBranch(lanes)
            	else:
            	    return self.construct_lane_tree(lanes, next_dimension, dimstop)
            else:
                left_branch = self.construct_lane_tree(left_branch_lanes, next_dimension, -1)
                right_branch = self.construct_lane_tree(right_branch_lanes, next_dimension, -1)
                return LaneTreeBinaryBranch(left_branch, right_branch)
        elif len(lanes) == 1:
            return lanes[0]
        else:
            return None

    def display(self, show_lanes: bool = True, block: bool = True):
        if show_lanes:
            for lane in list(self.id_lane_dict.values()):
                left_boundary_coord_array = lane.get_left_boundary_coord_array()
                right_boundary_coord_array = lane.get_right_boundary_coord_array()
                plt.plot(left_boundary_coord_array[0], left_boundary_coord_array[1], 'r', linewidth=1)
                plt.plot(right_boundary_coord_array[0], right_boundary_coord_array[1], 'g', linewidth=1)

        plt.gca().set_aspect('equal', adjustable='box')
        plt.show(block=block)

    def get_encapsulating_lane(self, aerial_coord: (float, float)) -> Lane:
        return self.lane_tree.get_encapsulating_lane(aerial_coord)

    def get_lane(self, global_id: str) -> Lane:
        return self.id_lane_dict.get(global_id)

    def get_lanes(self) -> [Lane]:
        return list(self.id_lane_dict.values())

    def get_traffic_light(self, global_id: str) -> TrafficLight:
        return self.id_traffic_light_dict.get(global_id)

    def get_traffic_lights(self) -> [TrafficLight]:
        return list(self.id_traffic_light_dict.values())

    def get_traffic_light_face(self, global_id: str) -> TrafficLightFace:
        return self.id_traffic_light_face_dict.get(global_id)

    def get_traffic_light_faces(self) -> [TrafficLightFace]:
        return list(self.id_traffic_light_face_dict.values())

    def get_parent_traffic_light(self, global_id: str) -> TrafficLight:
        return self.get_traffic_light(self.id_traffic_light_face_id_traffic_light_dict.get(global_id))

    def get_traffic_light_face_fluent(self, global_id: str, timestamp: float) -> fluent.TrafficLightFaceFluent:
        traffic_light_face_frames = self.id_traffic_light_face_frames_dict.get(global_id)
        if traffic_light_face_frames is None:
            return fluent.TrafficLightFaceFluent.UNKNOWN
        traffic_light_face_frames_pretimestamp = \
            [traffic_light_face_frame for traffic_light_face_frame in traffic_light_face_frames if traffic_light_face_frame[0] <= timestamp]
        if len(traffic_light_face_frames_pretimestamp) == 0:
            return fluent.TrafficLightFaceFluent.UNKNOWN
        else:
            return traffic_light_face_frames_pretimestamp[-1][1]

    def get_traffic_light_face_frames(self, global_id: str) -> [(float, fluent.TrafficLightFaceFluent)]:
        return self.id_traffic_light_face_frames_dict.get(global_id)

    def get_traffic_light_timestamps(self, global_id: str) -> [float]:
        traffic_light_timestamps = self.id_traffic_light_timestamps.get(global_id)
        return traffic_light_timestamps if traffic_light_timestamps is not None else []
