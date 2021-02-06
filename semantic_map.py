
from __future__ import annotations

import copy
import math
import itertools
import shapely.geometry
import numpy as np
import zarr
import l5kit.data.proto.road_network_pb2 as rnpb
import matplotlib.pyplot as plt

import fluent

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

def convert_local_enu_to_world(local_enu_coord_array: np.ndarray, local_enu_frame_geodetic_datum: np.ndarray, local_enu_frame_bearing_degrees: float, ecef_to_world: np.ndarray, equatorial_radius: float = 6.378137e6, polar_radius: float = 6.356752e6) -> (np.ndarray, np.ndarray):
    # https://en.wikipedia.org/wiki/Geographic_coordinate_conversion
    lat = local_enu_frame_geodetic_datum[0] * np.pi / 180
    lng = local_enu_frame_geodetic_datum[1] * np.pi / 180
    alt = local_enu_frame_geodetic_datum[2]

    centre_to_datum_unit = np.array([np.cos(lng) * np.cos(lat), np.sin(lng) * np.cos(lat), np.sin(lat)])
    up = centre_to_datum_unit
    east = np.cross(np.array([0, 0, 1]), up)
    east /= np.linalg.norm(east)
    north = np.cross(up, east)

    local_enu_frame_bearing_radians = local_enu_frame_bearing_degrees * np.pi / 180
    ox = np.cos(local_enu_frame_bearing_radians) * east - np.sin(local_enu_frame_bearing_radians) * north
    oy = np.sin(local_enu_frame_bearing_radians) * east + np.cos(local_enu_frame_bearing_radians) * north
    oz = up
    o = np.array([ox, oy, oz])

    excentricity = 1 - polar_radius**2 / equatorial_radius**2

    normal = equatorial_radius / np.sqrt(1 - excentricity * np.sin(lat)**2)
    origin = np.array([up[0] * normal, up[1] * normal, up[2] * (normal * (1 - excentricity))]) + alt * up

    ecef_coord_array = (origin + (o @ local_enu_coord_array).T).T

    homogeneous_ecef_coord_array = np.concatenate((ecef_coord_array, np.ones((1, ecef_coord_array.shape[-1]))))
    homogeneous_world_coord_array = ecef_to_world @ homogeneous_ecef_coord_array
    world_coord_array = homogeneous_world_coord_array[:3]

    world_o = ecef_to_world[:3,:3] @ o
    assert o[0,0] != 0 and o[2,2] != 0
    world_y = np.arctan(world_o[1,0] / world_o[0,0])
    world_p = np.arctan(-world_o[2,0] / np.sqrt(world_o[2,1]**2 + world_o[2,2]**2))
    world_r = np.arctan(world_o[2,1] / world_o[2,2])
    world_rpy = np.array([world_r, world_p, world_y])

    return world_coord_array, world_rpy

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

    def get_aerial_centroid(self) -> np.ndarray:
        raise Exception("This is an interface class not meant to be called")

    def get_coord_count(self) -> int:
        raise Exception("This is an interface class not meant to be called")

    def get_aerial_bb(self) -> (float, float, float, float):
        raise Exception("This is an interface class not meant to be called")

    def get_encapsulating_lane(self, aerial_coord: (float, float)) -> Lane:
        raise Exception("This is an interface class not meant to be called")

class Lane(LaneTreeInterface):

    def __init__(self, global_id: str, lane_data: rnpb.Lane, ecef_to_world: np.ndarray, semantic_map: SemanticMap):
        self.global_id = global_id
        self.semantic_map = semantic_map

        local_enu_frame_geodetic_datum = decompress_geoframe_origin(lane_data.geo_frame)
        local_enu_frame_bearing_degrees = lane_data.geo_frame.bearing_degrees

        self.left_boundary_coord_array, _ = convert_local_enu_to_world(
            convert_rel_to_abs(0.01 * np.stack((
                np.asarray(lane_data.left_boundary.vertex_deltas_x_cm),
                np.asarray(lane_data.left_boundary.vertex_deltas_y_cm),
                np.asarray(lane_data.left_boundary.vertex_deltas_z_cm)))),
            local_enu_frame_geodetic_datum,
            local_enu_frame_bearing_degrees,
            ecef_to_world)
        self.left_boundary_coord_array = np.float64(self.left_boundary_coord_array)
        self.left_boundary_aerial_centroid = np.mean(self.left_boundary_coord_array[:2], axis=-1)
        self.left_boundary_aerial_bb = (np.min(self.left_boundary_coord_array[0]),
                                        np.min(self.left_boundary_coord_array[1]),
                                        np.max(self.left_boundary_coord_array[0]),
                                        np.max(self.left_boundary_coord_array[1]))

        self.right_boundary_coord_array, _ = convert_local_enu_to_world(
            convert_rel_to_abs(0.01 * np.stack((
                np.asarray(lane_data.right_boundary.vertex_deltas_x_cm),
                np.asarray(lane_data.right_boundary.vertex_deltas_y_cm),
                np.asarray(lane_data.right_boundary.vertex_deltas_z_cm)))),
            local_enu_frame_geodetic_datum,
            local_enu_frame_bearing_degrees,
            ecef_to_world)
        self.right_boundary_coord_array = np.float64(self.right_boundary_coord_array)
        self.right_boundary_aerial_centroid = np.mean(self.right_boundary_coord_array[:2], axis=-1)
        self.right_boundary_aerial_bb = (np.min(self.right_boundary_coord_array[0]),
                                        np.min(self.right_boundary_coord_array[1]),
                                        np.max(self.right_boundary_coord_array[0]),
                                        np.max(self.right_boundary_coord_array[1]))

        self.coord_count = self.left_boundary_coord_array.shape[-1] + self.right_boundary_coord_array.shape[-1]
        self.aerial_centroid = (self.left_boundary_coord_array.shape[-1] * self.left_boundary_aerial_centroid +
            self.right_boundary_coord_array.shape[-1] * self.right_boundary_aerial_centroid) / self.coord_count
        self.aerial_bb = (min(self.left_boundary_aerial_bb[0], self.right_boundary_aerial_bb[0]),
            min(self.left_boundary_aerial_bb[1], self.right_boundary_aerial_bb[1]),
            max(self.left_boundary_aerial_bb[2], self.right_boundary_aerial_bb[2]),
            max(self.left_boundary_aerial_bb[3], self.right_boundary_aerial_bb[3]))

        self.access_restriction = lane_data.access_restriction
        self.adjacent_left_id = decode_global_id(lane_data.adjacent_lane_change_left)
        self.adjacent_right_id = decode_global_id(lane_data.adjacent_lane_change_right)
        self.ahead_ids = map(decode_global_id, lane_data.lanes_ahead)
        self.behind_ids = []
        self.traffic_control_ids = map(decode_global_id, lane_data.traffic_controls)

        # Uncomment to carry out sanity check by checking a path drawn between boundaries is within the lane. Currently broken due to weird lane structures.
        #larger_boundary_array, smaller_boundary_array = (self.left_boundary_coord_array[:2], self.right_boundary_coord_array[:2]) \
        #    if self.left_boundary_coord_array.shape[-1] >= self.right_boundary_coord_array.shape[-1] \
        #    else (self.right_boundary_coord_array[:2], self.left_boundary_coord_array[:2])
        #if larger_boundary_array.shape[-1] > 1:
        #    middle_path_coord_array = np.array(list(map(lambda i : larger_boundary_array[:,i] +
        #        np.float64(0.5)world_ * get_coord_min_projection(smaller_boundary_array, larger_boundary_array[:,i])[0],
        #        range(larger_boundary_array.shape[-1])))).T
        #    middle_path_coord_array = 0.5 * (middle_path_coord_array[:,:-1] + middle_path_coord_array[:,1:])
        #    assert all(map(lambda i : self.get_encapsulating_lane((middle_path_coord_array[0, i], middle_path_coord_array[1, i])), range(middle_path_coord_array.shape[-1])))

    def get_global_id(self) -> str:
        return self.global_id

    def get_aerial_centroid(self) -> np.ndarray:
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
            #lane_polygon_plot_x, lane_polygon_plot_y = lane_polygon.exterior.xy
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
                #lane_polygon_plot_x, lane_polygon_plot_y = lane_polygon.exterior.xy
                #plt.plot(self.left_boundary_coord_array[0], self.left_boundary_coord_array[1], 'r', linewidth=3)
                #plt.plot(self.right_boundary_coord_array[0], self.right_boundary_coord_array[1], 'g', linewidth=3)
                #plt.plot(lane_polygon_plot_x, lane_polygon_plot_y)
                #plt.plot(point.x, point.y, "yo")
                #plt.gca().set_aspect('equal', adjustable='box')
                #plt.show()
                return None
        else:
            return None

    def get_access_restriction(self) -> rpdb.AccessRestriction.Type:
        return self.access_restriction

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


class LaneTreeBinaryBranch(LaneTreeInterface):

    def __init__(self, left_branch, right_branch):
        self.left_branch = left_branch
        self.right_branch = right_branch

        self.coord_count = self.left_branch.get_coord_count() + self.right_branch.get_coord_count()
        self.aerial_centroid = (self.left_branch.get_coord_count() * self.left_branch.get_aerial_centroid() + self.right_branch.get_coord_count() * self.right_branch.get_aerial_centroid()) / self.coord_count
        left_aerial_bb = self.left_branch.get_aerial_bb()
        right_aerial_bb = self.right_branch.get_aerial_bb()
        self.aerial_bb = (min(left_aerial_bb[0], right_aerial_bb[0]),
            min(left_aerial_bb[1], right_aerial_bb[1]),
            max(left_aerial_bb[2], right_aerial_bb[2]),
            max(left_aerial_bb[3], right_aerial_bb[3]))

    def get_aerial_centroid(self) -> np.ndarray:
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


class LaneTreeKBranch(LaneTreeInterface):

    def __init__(self, branches):
        self.branches = branches

        self.coord_count = sum(map(lambda branch : branch.get_coord_count(), self.branches))
        self.aerial_centroid = sum(map(lambda branch : branch.get_coord_count() * branch.get_aerial_centroid(), self.branches)) / self.coord_count
        aerial_bbs = np.array(list(map(lambda branch : branch.get_aerial_bb(), self.branches)))
        self.aerial_bb = (min(aerial_bbs[:,0]),
            min(aerial_bbs[:,1]),
            max(aerial_bbs[:,2]),
            max(aerial_bbs[:,3]))

    def get_aerial_centroid(self) -> np.ndarray:
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


class TrafficLight:

    def __init__(self, global_id: str, traffic_control_element_data: rnpb.TrafficControlElement, ecef_to_world: np.ndarray, semantic_map: SemanticMap):
        self.global_id = global_id
        self.semantic_map = semantic_map

        assert traffic_control_element_data.geometry_type == rnpb.TrafficControlElement.GeometryType.UKNOWN

        local_enu_frame_geodetic_datum = decompress_geoframe_origin(traffic_control_element_data.geo_frame)
        local_enu_frame_bearing_degrees = traffic_control_element_data.geo_frame.bearing_degrees

        self.position, self.rpy = convert_local_enu_to_world(
            np.zeros((3, 1)),
            local_enu_frame_geodetic_datum,
            local_enu_frame_bearing_degrees,
            ecef_to_world)

        self.traffic_light_face_ids = map(decode_global_id, traffic_control_element_data.traffic_light.face_states)

    def get_global_id(self) -> str:
        return self.global_id

    def get_position(self) -> np.ndarray:
        return self.position

    def get_traffic_light_faces(self) -> [TrafficLightFace]:
        return list(map(self.semantic_map.get_traffic_light_face, self.traffic_light_face_ids))

    def get_active_traffic_light_face(self, timestamp: float) -> TrafficLightFace:
        traffic_light_face_fluents = map(lambda global_id : self.semantic_map.get_traffic_light_face_fluent(global_id, timestamp), self.traffic_light_face_ids)
        active_traffic_light_faces = [traffic_light_face_fluent_pair[0] for traffic_light_face_fluent_pair in zip(self.get_traffic_light_faces(), traffic_light_face_fluents) if traffic_light_face_fluent_pair[1] == fluent.TrafficLightFaceFluent.ACTIVE]
        if len(active_traffic_light_faces) == 0:
            return None
        else:
            return active_traffic_light_faces[0]

    def get_traffic_light_fluent_changes(self) -> [(float, fluent.TrafficLightFluent, fluent.TrafficLightFluent)]:
        traffic_light_timestamps = self.semantic_map.get_traffic_light_timestamps(self.get_global_id())
        active_traffic_light_faces = map(self.get_active_traffic_light_face, traffic_light_timestamps)
        active_traffic_light_face_types = map(lambda traffic_light_face : traffic_light_face.get_type(), active_traffic_light_faces)
        traffic_light_fluents = map(lambda traffic_light_face_type : TL_FACE_TYPE_TL_FLUENT_DICT.get(traffic_light_face_type, default=fluent.TrafficLightFluent.UNKNOWN), active_traffic_light_face_types)
        traffic_light_frames = zip(traffic_light_timestamps, active_traffic_light_faces)

        fluent_changes = []
        previous_fluent = fluent.TrafficLightFluent.UNKNOWN
        for current_frame in traffic_light_frames:
            if current_frame[1] != previous_fluent:
                fluent_changes.append((current_frame[0], previous_fluent, current_frame[1]))

            previous_fluent = current_frame[1]

        return fluent_changes


class TrafficLightFace:

    def __init__(self, global_id: str, traffic_control_element_data: rnpb.TrafficControlElement, traffic_light_face_type: str, semantic_map: SemanticMap):
        self.global_id = global_id
        self.semantic_map = semantic_map

        traffic_light_face_data = getattr(traffic_control_element_data, traffic_light_face_type)

        self.traffic_light_face_type = traffic_light_face_type
        self.observing_lane_ids = [decode_global_id(yield_rule.lane) for yield_rule in traffic_light_face_data.yield_rules_when_on]
        self.observing_lane_id_yield_to_lane_ids_dict = {decode_global_id(yield_rule.lane): map(decode_global_id, yield_rule.yield_to_lanes) for yield_rule in traffic_light_face_data.yield_rules_when_on}
        self.observing_lane_id_yield_to_crosswalk_ids_dict = {decode_global_id(yield_rule.lane): map(decode_global_id, yield_rule.yield_to_crosswalks) for yield_rule in traffic_light_face_data.yield_rules_when_on}
        self.no_right_turn = traffic_light_face_data.no_right_turn_on_red

        assert not self.no_right_turn or traffic_light_face_type.find("red") >= 0

    def get_global_id(self) -> str:
        return self.global_id

    def get_type(self) -> str:
        return self.traffic_light_face_type

    def get_observing_lanes(self) -> [Lane]:
        return map(self.semantic_map.get_lane, self.observing_lane_id)

    def get_yield_to_lanes_for_observing_lane(self, lane: Lane) -> [Lane]:
        return map(self.semantic_map.get_lane, self.observing_lane_id_yield_to_lane_ids_dict.get(lane.get_global_id()))

    def get_traffic_light_face_fluent_changes(self) -> [(float, fluent.TrafficLightFaceFluent, fluent.TrafficLightFaceFluent)]:
        traffic_light_face_frames = self.semantic_map.get_traffic_light_face_frames(self.get_global_id())

        fluent_changes = []
        previous_fluent = fluent.TrafficLightFaceFluent.UNKNOWN
        for current_frame in traffic_light_face_frames:
            if current_frame[1] != previous_fluent:
                fluent_changes.append((current_frame[0], previous_fluent, current_frame[1]))

            previous_fluent = current_frame[1]

        return fluent_changes


class SemanticMap:

    def __init__(self, map_pb_file_path: str, scene_zarr_dir_path: str, ecef_to_world: np.ndarray):
        with open(map_pb_file_path, "rb") as map_pb_file:
            map_fragment = rnpb.MapFragment()
            map_fragment.ParseFromString(map_pb_file.read())

        scene_dataset = zarr.open(scene_zarr_dir_path)

        frame_data = scene_dataset["frames"]
        tl_face_data = scene_dataset["traffic_light_faces"]

        self.id_traffic_light_face_frames_dict = {}
        self.id_traffic_light_timestamps = {}

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

            if previous_timestamp is not None:
                frame_deltatime_sum += current_frame_data["timestamp"] / 1e9 - previous_timestamp

            previous_timestamp = current_frame_data["timestamp"] / 1e9

            counter += 1
        frame_deltatime_mean = frame_deltatime_sum / counter
        print("")
        print("Finished processing frames")

        self.lanes = []
        self.id_lane_dict = {}
        self.traffic_lights = []
        self.id_traffic_light_dict = {}
        self.traffic_light_faces = []
        self.id_traffic_light_face_dict = {}

        counter = 0
        for element in map_fragment.elements:
            print("Processed {0:.2f}% of map elements".format(100 * counter / len(map_fragment.elements)), end="\r")

            global_id = decode_global_id(element.id)

            if bool(element.element.HasField("lane")):
                lane = Lane(global_id, element.element.lane, ecef_to_world, self)
                self.id_lane_dict[global_id] = lane
                self.lanes.append(lane)

            if bool(element.element.HasField("traffic_control_element")):
                traffic_control_element_type = element.element.traffic_control_element.WhichOneof("Type")

                if traffic_control_element_type == "traffic_light":
                    traffic_light = TrafficLight(global_id, element.element.traffic_control_element, ecef_to_world, self)
                    self.id_traffic_light_dict[global_id] = traffic_light
                    self.traffic_lights.append(traffic_light)

                if traffic_control_element_type in TL_FACE_TYPE_TL_FLUENT_DICT.keys():
                    traffic_light_face = TrafficLightFace(global_id, element.element.traffic_control_element, traffic_control_element_type, self)
                    self.id_traffic_light_face_dict[global_id] = traffic_light_face
                    self.traffic_light_faces.append(traffic_light_face)

            counter += 1
        print("")
        print("Finished processing map elements")

        counter = 0
        for lane in self.lanes:
            print("Post-processed {0:.2f}% of lanes".format(100 * counter / len(self.lanes)), end="\r")

            lane_global_id = lane.get_global_id()

            lanes_ahead = lane.get_lanes_ahead()
            for lane_ahead in lanes_ahead:
                lane_ahead.append_behind_id(lane_global_id)

            counter += 1
        print("")
        print("Finished post-processing lanes")

        counter = 0
        for traffic_light_face in self.traffic_light_faces:
            print("Post-processed {0:.2f}% of traffic light faces".format(100 * counter / len(self.traffic_light_faces)), end="\r")

            traffic_light_face_frames = self.id_traffic_light_face_frames_dict.get(traffic_light_face.get_global_id())
            if traffic_light_face_frames is None:
                self.id_traffic_light_face_frames_dict[traffic_light_face.get_global_id()] = [(0, fluent.TrafficLightFaceFluent.UNKNOWN)]
            else:
                if traffic_light_face_frames[-1][1] != fluent.TrafficLightFaceFluent.UNKNOWN:
                    self.id_traffic_light_face_frames_dict[traffic_light_face.get_global_id()].append(
                        (traffic_light_face_frames[-1][0] + frame_deltatime_mean, fluent.TrafficLightFaceFluent.UNKNOWN))

            counter += 1
        print("")
        print("Finished post-processing traffic light faces")

        self.lane_tree = self.construct_lane_tree(self.lanes)
        print("Finished constructing lane tree")

    def construct_lane_tree(self, lanes: [Lane], dimension: int = 0, dimstop: int = -1) -> LaneTreeInterface:
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

    def get_encapsulating_lane(self, aerial_coord: (float, float)) -> Lane:
        return self.lane_tree.get_encapsulating_lane(aerial_coord)

    def get_lane(self, global_id: str) -> Lane:
        return self.id_lane_dict.get(global_id)

    def get_traffic_light(self, global_id: str) -> TrafficLight:
        return self.id_traffic_light_dict.get(global_id)

    def get_traffic_light_face(self, global_id: str) -> TrafficLightFace:
        return self.id_traffic_light_face_dict.get(global_id)

    def get_traffic_light_face_fluent(self, global_id: str, timestamp: float) -> fluent.TrafficLightFaceFluent:
        traffic_light_face_frames = self.id_traffic_light_face_frames_dict.get(global_id)
        if traffic_light_face_frames is None:
            return fluent.TrafficLightFaceFluent.UNKNOWN
        traffic_light_face_frames_pretimestamp = \
            [traffic_light_face_frame for traffic_light_face_frame in traffic_light_face_frames if traffic_light_face_frame[0] <= timestamp]
        return traffic_light_face_frames_pretimestamp[-1][1]

    def get_traffic_light_face_frames(self, global_id: str) -> [(float, fluent.TrafficLightFaceFluent)]:
        return self.id_traffic_light_face_frames_dict.get(global_id)

    def get_traffic_light_timestamps(self, global_id: str) -> [float]:
        traffic_light_timestamps = self.id_traffic_light_timestamps.get(global_id)
        return traffic_light_timestamps if traffic_light_timestamps is not None else []
