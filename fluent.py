
import math
import enum
import copy
import numpy as np

class Fluent(enum.IntEnum):

    @staticmethod
    def get_affinity_matrix():
        return None

    def get_affinity(self, fluent):
        affinity_matrix = get_affinity_matrix()

        if affinity_matrix is None:
            raise Exception("There is no affinity matrix")

        if affinity_matrix.shape[0] <= self or affinity_matrix.shape[0] <= fluent:
            raise Exception("Index pair outside of affinity matrix bounds")

        return affinity_matrix[self, fluent]


class TrafficLightFluent(Fluent):

    @staticmethod
    def get_affinity_matrix():
        return np.array([[0, 1, 2, 3], [1, 0, 2, 3], [1, 2, 0, 3], [1, 3, 2, 0]])

    UNKNOWN = -1
    RED = 0
    YELLOW = 1
    GREEN = 2


class TrafficLightFaceFluent(Fluent):

    @staticmethod
    def get_affinity_matrix():
        return np.array([[0, 1, 2], [1, 0, 2], [1, 2, 0]])

    UNKNOWN = -1
    INACTIVE = 0
    ACTIVE = 1


class MovementFluent(Fluent):

    @staticmethod
    def get_affinity_matrix():
        return np.array([[0, 1, 2, 3, 4], [1, 0, 4, 3, 2], [1, 4, 0, 2, 3], [1, 3, 2, 0, 4], [1, 2, 3, 4, 0]])

    UNKNOWN = -1
    STATIONARY = 0
    MOVING_CONSTANT = 1
    MOVING_ACCELERATING = 2
    MOVING_DECELERATING = 3

def calculate_heuristic(fluent_changes: [(float, MovementFluent, MovementFluent)], fluent_stability_window_threshold: float):
    unstable_fluent_change_index_pairs = get_unstable_fluent_change_index_pairs(fluent_changes, fluent_stability_window_threshold)
    timespans = map(lambda unstable_fluent_change_index_pair : fluent_changes[unstable_fluent_change_index_pair[1]][0] - fluent_changes[unstable_fluent_change_index_pair[0]][0], unstable_fluent_change_index_pairs)
    min_timespan = min(timespans)
    if len(unstable_fluent_change_index_pairs) == 1:
        heuristic = min_timespan
    else:
        heuristic = min_timespan * math.floor(len(unstable_fluent_change_index_pairs) / 2)
    return heuristic, unstable_fluent_change_index_pairs

def calculate_time_changed(fluent_changes: [(float, MovementFluent, MovementFluent)], new_fluent_changes: [(float, MovementFluent, MovementFluent)]):
    cumulative_time_changed = 0.0

    if len(fluent_changes) == 0 or len(new_fluent_changes) == 0:
        raise ValueError("Invalid old or new fluent change list length. Old Length: {}, New Length: {}".format(len(fluent_changes), len(new_fluent_changes)))

    if fluent_changes[0][0] != new_fluent_changes[0][0]:
        raise ValueError("First timestamp mismatch between old and new fluent change list. Old First Timestamp: {}, New First Timestamp: {}".format(fluent_changes[0][0], new_fluent_changes[0][0]))

    if fluent_changes[-1][0] != new_fluent_changes[-1][0]:
        raise ValueError("Final timestamp mismatch between old and new fluent change list. Old Final Timestamp: {}, New Final Timestamp: {}".format(fluent_changes[-1][0], new_fluent_changes[-1][0]))

    i = 0
    j = 0
    while i < len(fluent_changes) or j < len(new_fluent_changes):

        pre_fluent = fluent_changes[i][1]
        post_fluent = fluent_changes[i][2]
        pre_new_fluent = new_fluent_changes[j][1]
        post_new_fluent = new_fluent_changes[j][2]

        if fluent_changes[i][0] == new_fluent_changes[j][0]:
            i += 1
        elif fluent_changes[i][0] > new_fluent_changes[j][0]:
            if post_new_fluent != pre_fluent:
                cumulative_time_changed += fluent_changes[i][0] - new_fluent_changes[j][0]
            j += 1
        else:
            if post_fluent != pre_new_fluent:
                cumulative_time_changed += new_fluent_changes[i][0] - fluent_changes[j][0]
            i += 1

    return cumulative_time_changed

def edit_fluent_changes(fluent_changes: [(float, MovementFluent, MovementFluent)], fluent_change_edits: [(int, int, MovementFluent)]):
    new_fluent_changes = copy.deepcopy(fluent_changes)
    mapping = { i: i for i in range(len(fluent_changes)) }

    for (start, finish, new_fluent) in fluent_change_edits:
        shift_size = (finish - start) - 1
        for i in range(finish, len(new_fluent_changes)):
            mapping[i - shift_size] = mapping[i]
            del mapping[i]

        start_timestamp = new_fluent_changes[start][0]
        start_fluent = new_fluent_changes[start][1]
        finish_timestamp = new_fluent_changes[finish][0]
        finish_fluent = new_fluent_changes[finish][2]

        for i in range(start, finish + 1):
            del new_fluent_changes[i]

        new_fluent_changes.insert(start, (finish_timestamp, new_fluent, finish_fluent))
        new_fluent_changes.insert(start, (start_timestamp, start_fluent, new_fluent))

    return new_fluent_changes, mapping

def get_unstable_fluent_change_index_pairs(fluent_changes: [(float, MovementFluent, MovementFluent)], fluent_stability_window_threshold: float):
    unstable_fluent_change_index_pairs = []

    for i in range(len(fluent_changes) - 1):
        if fluent_changes[i + 1][0] - fluent_changes[i][0] < fluent_stability_window_threshold:
            unstable_fluent_change_index_pairs.append((i, i + 1))

    return unstable_fluent_change_index_pairs

def get_fluent_change_edit_from_index_pair(fluent_changes: [(float, MovementFluent, MovementFluent)], unstable_fluent_change_index_pair: (int, int)):
    fluent_to_edit = fluent_changes[unstable_fluent_change_index_pair[0]][2]
    if fluent_to_edit != fluent_changes[unstable_fluent_change_index_pair[1]][1]:
        raise ValueError("Either fluent changes or index pair are invalid, fluent to edit is not consistent. {} -> ({} vs {}) -> {}".format(fluent_changes[unstable_fluent_change_index_pair[0]][1], fluent_changes[unstable_fluent_change_index_pair[0]][2], fluent_changes[unstable_fluent_change_index_pair[1]][1], fluent_changes[unstable_fluent_change_index_pair[1]][2]))

    fluent_before = fluent_changes[unstable_fluent_change_index_pair[0]][1]
    fluent_after = fluent_changes[unstable_fluent_change_index_pair[1]][2]

    if fluent_before == fluent_after:
        return (*unstable_fluent_change_index_pair, fluent_before)
    else:
        fluent_before_affinity = fluent_to_edit.get_affinity(fluent_before)
        fluent_after_affinity = fluent_to_edit.get_affinity(fluent_after)
         # NOTE: If affinity matrix is defined correctly, it should be impossible for the two affinities to be equal at this point
        if fluent_before_affinity <= fluent_after_affinity:
            return (*unstable_fluent_change_index_pair, fluent_before)
        else:
            return (*unstable_fluent_change_index_pair, fluent_after)

def explore_fluent_change_edits(fluent_changes: [(float, MovementFluent, MovementFluent)], new_fluent_changes: [(float, MovementFluent, MovementFluent)], unstable_fluent_change_index_pairs: [(int, int)], fluent_stability_window_threshold: float):
    discovered_fluent_change_edits = map(lambda unstable_fluent_change_index_pair : get_fluent_change_edit_from_index_pair(new_fluent_changes, unstable_fluent_change_index_pair), unstable_fluent_change_index_pairs)
    discovered_new_fluent_changes_list, discovered_mappings = zip(*map(lambda discovered_fluent_change_edit : edit_fluent_changes(new_fluent_changes, discovered_fluent_change_edit), discovered_fluent_change_edits))
    discovered_time_changed_list = map(lambda discovered_new_fluent_changes : calculate_time_changed(fluent_changes, discovered_new_fluent_changes), discovered_new_fluent_changes_list)
    discovered_time_heuristics, discovered_new_unstable_fluent_change_index_pairs_list = zip(*map(lambda discovered_new_fluent_changes : calculate_heuristic(discovered_new_fluent_changes, fluent_stability_window_threshold)))
