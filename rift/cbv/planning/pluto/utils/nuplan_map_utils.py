#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : nuplan_map_utils.py
@Date    : 2024/10/02
"""
import os
from pathlib import Path
import shapely

import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon, MultiLineString, GeometryCollection
from shapely.ops import nearest_points
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, List

from rift.cbv.planning.pluto.feature_builder.common import interpolate_polyline
from nuplan_plugin.observation.utils import (
    create_path_from_se2,
    path_to_linestring,
)
from rift.cbv.planning.pluto.utils.nuplan_state_utils import CarlaAgentState, get_sample_ego_state
from nuplan_plugin.actor_state.state_representation import StateSE2
from nuplan_plugin.maps.maps_datatypes import SemanticMapLayer
from nuplan_plugin.path.utils import trim_path
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)



class CarlaMapObject:
    """
    A class representing a map object with attributes accessible via dot notation.
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)



class CarlaMap:
    """
    Carla HD map API class
    """
    def __init__(
            self,
            carla_town_name: str,
            map_sample_points: int = 20,
            map_data_dir: str = 'data/map_data',
            speed_limit_mps: int = 14,  # m/s
    ):
        self.map_sample_points = map_sample_points
        self.speed_limit_mps = speed_limit_mps
        filename = os.path.join(map_data_dir, carla_town_name + '_HD_map.npz')
        
        # load and pre-process data
        map_data_dict = self._load_hd_map(filename)

        self.lane_gdf, self.lane_connector_gdf, self.all_lane_object_gdf, self.crosswalk_gdf = self._preprocess_data(map_data_dict)
        
        # create spatial index
        self.lane_sindex = self.lane_gdf.sindex
        self.lane_connector_sindex = self.lane_connector_gdf.sindex
        self.all_lane_object_sindex  = self.all_lane_object_gdf.sindex
        self.crosswalk_sindex = self.crosswalk_gdf.sindex

    def _load_hd_map(self, filename: str) -> Dict[Any, Any]:
            """
            Load the HD-Map data from a .npz file.

            Parameters:
            - filename: Full path to the map data file.

            Returns:
            - data_dict: A dictionary containing the map data.
            """
            data = np.load(filename, allow_pickle=True)
            data_dict = dict(data['arr'])
            return data_dict
    

    def _get_geometry(self, coords: np.ndarray, threshold: float = 20):
        """
        Generate a geometry object based on the input coordinate array.

        Parameters:
        coords (numpy.ndarray): A coordinate array of shape (N, 2).
        threshold (float): Distance threshold to determine line segment continuity.

        Returns:
        Geometry: A geometry object (Point/LineString/MultiLineString/empty GeometryCollection).
        """
        # Handle empty array
        if len(coords) == 0:
            return GeometryCollection()  # Return an empty geometry object
        
        # Handle single point case
        elif len(coords) == 1:
            return Point(coords[0])
        
        # Handle multiple points case
        else:
            threshold_sq = threshold ** 2  # Use squared distance to avoid square root computation
            dx = np.diff(coords[:, 0])  # Differences in x-coordinates
            dy = np.diff(coords[:, 1])  # Differences in y-coordinates
            squared_distances = dx * dx + dy * dy  # Squared distances between consecutive points
            
            # Find indices where the distance exceeds the threshold
            split_indices = np.where(squared_distances > threshold_sq)[0]
            
            # Generate split points (including start and end indices)
            split_points = np.concatenate([[0], split_indices + 1, [len(coords)]])
            
            # Extract continuous line segments
            lines = []
            for i in range(len(split_points) - 1):
                start = split_points[i]  # Start index of the segment
                end = split_points[i + 1]  # End index of the segment
                if end - start >= 2:  # Ensure the segment has at least two points
                    lines.append(LineString(coords[start:end]))
            
            # Return geometry based on the number of segments
            if len(lines) == 1:
                return lines[0]  # Single LineString
            elif lines:
                return MultiLineString(lines)  # Multiple LineStrings
            else:
                return MultiLineString()  # Empty MultiLineString if no valid segments are found


    def _preprocess_data(self, data_dict: Dict[Any, Any]) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Preprocess the map data and convert it into GeoDataFrames for spatial queries.

        Parameters:
        - data_dict: A dictionary containing the raw map data.

        Returns:
        - lane_gdf: GeoDataFrame of Lanes.
        - lane_connector_gdf: GeoDataFrame of LaneConnectors.
        - crosswalk_gdf: GeoDataFrame of Crosswalks.
        """
        all_lane_object_features = []
        lane_features = []
        lane_connector_features = []
        crosswalk_features = []
        token_id = 0

        # Process LaneMarks
        for road_id, road_data in data_dict.items():
            if road_id == 'Crosswalks':
                continue  # Will be processed later

            # Process each lane
            for lane_id, lane_data in road_data.items():
                if lane_id == 'Trigger_Volumes':
                    continue  # Skip Trigger_Volumes

                lane_type = lane_data.get('LaneType', None)
                lane_width = lane_data.get('LaneWidth', 3.5)  # Default lane width is 4.0 meters
                lane_marks_dict = lane_data.get('LaneMark', {})

                for side in ['Left', 'Center', 'Right']:
                    lane_mark_list = lane_marks_dict.get(side, [])

                    if len(lane_mark_list) > 1:
                        # merge multi lane mark
                        merged_lane_mark = lane_mark_list[0].copy()
                        for lm in lane_mark_list[1:]:
                            # merge 'Points'
                            merged_lane_mark['Points'].extend(lm['Points'])
                            merged_lane_mark['Topology'].extend(lm['Topology'])
                        lane_marks_dict[side] = merged_lane_mark
                    elif len(lane_mark_list) == 1:
                        lane_marks_dict[side] = lane_mark_list[0]
                    else:
                        lane_marks_dict[side] = None

                # process coordinates
                geometry = {'Left': None, 'Center': None, 'Right': None}
                state_list = {'Left': None, 'Center': None, 'Right': None}
                coords_list = {'Left': None, 'Center': None, 'Right': None}
                for side in ['Left', 'Center', 'Right']:
                    lane_marks = lane_marks_dict.get(side, None)
                    if lane_marks is None:
                        continue

                    points_data = lane_marks['Points']
                    state_list[side] = [None] * len(points_data)
                    coords_list[side] = np.zeros((len(points_data), 2))
                    for i, point_info in enumerate(points_data):
                        x, y, _ = point_info[0]  # ((x, y, z), (roll, pitch, yaw))
                        _, _, yaw = point_info[1]

                        state_list[side][i] = StateSE2(x=x, y=y, heading=yaw)
                        coords_list[side][i] = (x, y)

                    geometry[side] = self._get_geometry(coords_list[side])
                
                center_coords = coords_list['Center']
                left_coords = coords_list['Left']
                right_coords = coords_list['Right']
                
                # # process left coords
                # if not state_list['Left']:
                #     directions = np.diff(center_coords, axis=0)
                #     normals = np.vstack((-directions[:, 1], directions[:, 0])).T
                #     normals /= np.linalg.norm(normals, axis=1, keepdims=True)
                #     left_coords = center_coords[1:] + lane_width / 2 * normals  # for each point, offset the normal vector
                # # process right coords
                # if not state_list['Right']:
                #     directions = np.diff(center_coords, axis=0)
                #     normals = np.vstack((-directions[:, 1], directions[:, 0])).T
                #     normals /= np.linalg.norm(normals, axis=1, keepdims=True)
                #     right_coords = center_coords[:-1] - lane_width / 2 * normals  # for each point, offset the normal vector
                
                # process polygon
                polygon_coords = np.vstack([left_coords, np.flipud(right_coords)])  # Stack and reverse right coords

                # Create the polygon if coordinates are available
                if polygon_coords is not None and polygon_coords.size > 0:
                    lane_polygon = Polygon(polygon_coords)
                else:
                    lane_polygon = None
                centerline, edges = self.build_lane_edges(left_coords, center_coords, right_coords)
                # Store features
                feature = {
                    'token_id': token_id,
                    'geometry': geometry['Center'],         # the geometry of the center line
                    'width': lane_width,                    # the width of the lane
                    'center_states': state_list['Center'],  # the descrite states of the center line
                    'edges': edges,                         # the edges of the lane
                    'centerline': centerline,               # the centerline of the lane
                    'center_coords': center_coords,         # the center coordinates of the lane in np form
                    'road_id': road_id,                     # road id
                    'lane_id': lane_id,                     # lane id
                    'lane_type': lane_type,                 # the type of the lane, can be 'Driving'
                    'mark_type': lane_marks_dict['Center']['Type'],  # center linemark type, 'Center'
                    'color': lane_marks_dict['Center']['Color'],  # color of the center mark 'White'
                    'topology': lane_marks_dict['Center']['Topology'],  # 'road_id' and 'lane_id' of the current road adjacent to, formed as ((road_id, lane_id), ..)
                    'topology_type': lane_marks_dict['Center']['TopologyType'],  # 'Junction' or 'Normal'
                    'left': lane_marks_dict['Center']['Left'],   # The road_id and lane_id of the left lane of the current lane，formed as (road_id, lane_id)
                    'right': lane_marks_dict['Center']['Right'],  # The road_id and lane_id of the right lane of the current lane，formed as (road_id, lane_id)
                    'polygon': lane_polygon,                # the polygon of the lane (for rendering)
                    'speed_limit_mps': 14,                  # the speed limit of the lane (14 mps) or (50 mph)
                    # Add other necessary attributes
                }
                token_id += 1
                all_lane_object_features.append(feature)
                if feature['topology_type'] == 'Normal':
                    lane_features.append(feature)
                elif feature['topology_type'] == 'Junction':
                    lane_connector_features.append(feature)
                else:
                    print(f"UnSupport topology type: {feature['topology_type']}")

        # Create GeoDataFrame for Lane and LaneConnector
        lane_gdf = gpd.GeoDataFrame(lane_features, geometry='geometry')
        lane_connector_gdf = gpd.GeoDataFrame(lane_connector_features, geometry='geometry')
        all_lane_object_gdf = gpd.GeoDataFrame(all_lane_object_features, geometry='geometry')

        # Process Crosswalks
        crosswalks_data = data_dict.get('Crosswalks', [])
        for crosswalk in crosswalks_data:
            polygon = crosswalk.get('Polygon', None)
            location = crosswalk.get('Location', None)

            if polygon:
                # Assume polygon is already a shapely.geometry.Polygon object
                feature = {
                    'token_id': token_id,
                    'geometry': polygon,
                    'polygon': polygon,
                    'location': location,
                    'edges': self._get_crosswalk_edges(polygon)
                    # Add other necessary attributes
                }
                token_id += 1
                crosswalk_features.append(feature)

        # Handle the case where there are no crosswalk features
        if not crosswalk_features:
            # Create an empty GeoDataFrame with the necessary columns
            crosswalk_gdf = gpd.GeoDataFrame(columns=['geometry', 'polygon', 'location'])
            crosswalk_gdf = crosswalk_gdf.set_geometry('geometry')
        else:
            crosswalk_gdf = gpd.GeoDataFrame(crosswalk_features, geometry='geometry')

        return lane_gdf, lane_connector_gdf, all_lane_object_gdf, crosswalk_gdf

    def build_lane_edges(self, left_coords: np.ndarray, center_coords: np.ndarray, right_coords: np.ndarray, sample_points=20):
        centerline = self._sample_discrete_path(
            center_coords, sample_points + 1
        )
        # sample for the left bound
        left_bound = self._sample_discrete_path(
            left_coords, sample_points + 1
        )
        # sample for the right bound
        right_bound = self._sample_discrete_path(
            right_coords, sample_points + 1
        )
        edges = np.stack([centerline, left_bound, right_bound], axis=0)
        return centerline, edges
    
    def _sample_discrete_path(self, discrete_path: np.ndarray, num_points: int):
        return interpolate_polyline(discrete_path, num_points)
    
    def _get_crosswalk_edges(
        self, crosswalk_polygon: Polygon, sample_points: int = 21
    ):
        bbox = shapely.minimum_rotated_rectangle(crosswalk_polygon)
        coords = np.stack(bbox.exterior.coords.xy, axis=-1)
        edge1 = coords[[3, 0]]  # right boundary
        edge2 = coords[[2, 1]]  # left boundary

        edges = np.stack([(edge1 + edge2) * 0.5, edge2, edge1], axis=0)  # [3, 2, 2]
        vector = edges[:, 1] - edges[:, 0]  # [3, 2]
        steps = np.linspace(0, 1, sample_points, endpoint=True)[None, :]
        points = edges[:, 0][:, None, :] + vector[:, None, :] * steps[:, :, None]

        return points

    def query_proximal_lane_data(self, point: Point, radius: float) -> List[CarlaMapObject]:
        """
        Query lane data within a specified radius of a given geographic coordinate.

        Parameters:
        - point: Geographic coordinate.
        - radius: Query radius, in the same unit as the coordinates.

        Returns:
        - proximal_lanes: List of CarlaMapObjects representing the proximal lanes.
        """
        proximal_lanes: List[CarlaMapObject] = []
        buffer = point.buffer(radius)

        possible_lane_index: List[int] = list(self.lane_sindex.intersection(buffer.bounds))
        possible_lanes: gpd.GeoDataFrame = self.lane_gdf.iloc[possible_lane_index]
        precise_lanes: gpd.GeoDataFrame = possible_lanes[possible_lanes.intersects(buffer)]

        if not precise_lanes.empty:
            precise_lanes = precise_lanes.copy()
            precise_lanes['distance'] = precise_lanes.geometry.distance(point)
            sorted_lanes = precise_lanes.sort_values('distance')

            for _, row in sorted_lanes.iterrows():
                obj = CarlaMapObject(**row.to_dict())
                proximal_lanes.append(obj)

        return proximal_lanes

    def query_proximal_map_data(self, point: Point, radius: float) -> Dict[SemanticMapLayer, List[CarlaMapObject]]:
        """
        Query map elements within a specified radius of a given geographic coordinate,
        and return them as a dictionary mapping SemanticMapLayer to lists of CarlaMapObjects.

        Parameters:
        - x, y: Geographic coordinates.
        - radius: Query radius, in the same unit as the coordinates.

        Returns:
        - proximal_map_data: Dictionary mapping SemanticMapLayer to lists of CarlaMapObjects.
        """

        proximal_map_data: Dict[SemanticMapLayer, List[CarlaMapObject]] = {}
        buffer = point.buffer(radius)

        # Query Lanes
        possible_lane_index: List[int] = list(self.lane_sindex.intersection(buffer.bounds))
        possible_lanes: gpd.GeoDataFrame = self.lane_gdf.iloc[possible_lane_index]
        precise_lanes: gpd.GeoDataFrame = possible_lanes[possible_lanes.intersects(buffer)]

        # Compute distances and sort LaneMarks
        if not precise_lanes.empty:
            precise_lanes = precise_lanes.copy()
            precise_lanes['distance'] = precise_lanes.geometry.distance(point)
            sorted_lanes = precise_lanes.sort_values('distance')

            # Convert to list of CarlaMapObjects
            lane_objects: List[CarlaMapObject] = []
            for _, row in sorted_lanes.iterrows():
                obj = CarlaMapObject(**row.to_dict())
                lane_objects.append(obj)

            proximal_map_data[SemanticMapLayer.LANE] = lane_objects
        else:
            proximal_map_data[SemanticMapLayer.LANE] = []

        # Query LaneConnectors
        possible_lane_connector_index: List[int] = list(self.lane_connector_sindex.intersection(buffer.bounds))
        possible_lane_connectors: gpd.GeoDataFrame = self.lane_connector_gdf.iloc[possible_lane_connector_index]
        precise_lane_connectors: gpd.GeoDataFrame = possible_lane_connectors[possible_lane_connectors.intersects(buffer)]

        # Compute distances and sort LaneMarks
        if not precise_lane_connectors.empty:
            precise_lane_connectors = precise_lane_connectors.copy()
            precise_lane_connectors['distance'] = precise_lane_connectors.geometry.distance(point)
            sorted_lane_connectors = precise_lane_connectors.sort_values('distance')

            # Convert to list of CarlaMapObjects
            lane_connector_objects: List[CarlaMapObject] = []
            for _, row in sorted_lane_connectors.iterrows():
                obj = CarlaMapObject(**row.to_dict())
                lane_connector_objects.append(obj)

            proximal_map_data[SemanticMapLayer.LANE_CONNECTOR] = lane_connector_objects
        else:
            proximal_map_data[SemanticMapLayer.LANE_CONNECTOR] = []

        # Query Crosswalks
        possible_crosswalks_index: List[int] = list(self.crosswalk_sindex.intersection(buffer.bounds))
        possible_crosswalks: gpd.GeoDataFrame = self.crosswalk_gdf.iloc[possible_crosswalks_index]
        precise_crosswalks: gpd.GeoDataFrame = possible_crosswalks[possible_crosswalks.intersects(buffer)]

        # Compute distances and sort Crosswalks
        if not precise_crosswalks.empty:
            precise_crosswalks = precise_crosswalks.copy()
            precise_crosswalks['distance'] = precise_crosswalks.geometry.distance(point)
            sorted_crosswalks = precise_crosswalks.sort_values('distance')

            # Convert to list of CarlaMapObjects
            crosswalk_objects: List[CarlaMapObject] = []
            for _, row in sorted_crosswalks.iterrows():
                obj = CarlaMapObject(**row.to_dict())
                crosswalk_objects.append(obj)

            proximal_map_data[SemanticMapLayer.CROSSWALK] = crosswalk_objects
        else:
            proximal_map_data[SemanticMapLayer.CROSSWALK] = []

        return proximal_map_data
    
    def find_nearest_point_on_geometry(self, geometry, target_point):
        """
        Find the nearest point on a geometric object (LineString/MultiLineString) 
        to a given target point, as well as the shortest distance.

        Args:
            geometry (LineString/MultiLineString): The input geometric object.
            target_point (Point): The target point.

        Returns:
            nearest_point (Point): The nearest point on the geometric object.
            min_distance (float): The shortest distance.
        """
        # Ensure the inputs are Shapely geometry objects
        if not isinstance(geometry, (Point, LineString, MultiLineString)):
            raise ValueError("geometry must be a Point, LineString, or MultiLineString")

        # Use nearest_points to find the closest pair of points
        # Returns two points: the first is on the target_point, the second is on the geometry
        _, nearest_on_geometry = nearest_points(target_point, geometry)
        
        # Calculate the shortest distance
        min_distance = target_point.distance(nearest_on_geometry)
        
        return nearest_on_geometry, min_distance

    def collect_adjacent_lanes(self, cur_state, base_lane, direction, max_attempts=5):
        """
        Collect adjacent lanes of the base lane in the given direction.
        """
        current_lane = base_lane
        start_point = Point(*cur_state.rear_axle.point.array)
        lanes = []
        
        for _ in range(max_attempts):
            adjacent_info = current_lane.get(direction, [])
            if not adjacent_info:
                break
            
            # if the adjacent lane is not in the same road, break
            if adjacent_info[0] != base_lane['road_id']:
                break
            
            # if the lane_id is not the same sign as the base lane and the center states is too short, break
            adj_lane = self.get_lane_by_id(*adjacent_info)
            if not adj_lane or adj_lane['lane_id'] * base_lane['lane_id'] <= 0 or len(adj_lane['center_states']) <= 2:
                break
            
            # if the distance between to adjacent lane is too large, break (highway entry/exit point bug)
            start_point, adj_dis = self.find_nearest_point_on_geometry(adj_lane['geometry'], start_point)
            if adj_dis >= adj_lane['width'] * 3:
                break
                
            lanes.append(adj_lane)
            current_lane = adj_lane
            
        return lanes

    def query_reference_lines(self, current_waypoints, current_state: CarlaAgentState, route_ids: Dict[str, List[int]], max_length: float=120):
        """
        Query all the possible reference lanes under given max_length.

        Parameters:
        - current_waypoints: the current waypoint list.
        - current_state: the current state of CBV
        - route_road_ids: the road id set of the CBV global route
        - max_length: the max length of the reference lane.

        Returns:
        - reference_lanes: List of LineStrings representing the reference lanes.
        """
        # find the current lane
        for current_waypoint in current_waypoints:
            current_road_id = current_waypoint.road_id
            current_lane_id = current_waypoint.lane_id
            nearest_lane = self.get_lane_by_id(current_road_id, current_lane_id)
            if nearest_lane is not None and len(nearest_lane['center_states']) > 2:
                break
            else:
                print(f">> lane obj with road_id:{current_road_id} lane_id:{current_lane_id} not exist or too short")

        start_lanes = [nearest_lane]

        start_lanes += self.collect_adjacent_lanes(current_state, nearest_lane, 'left')
        start_lanes += self.collect_adjacent_lanes(current_state, nearest_lane, 'right')

        # traverse topology for each start lane
        candidate_route = []
        route_dict = set(zip(route_ids['road_ids'], route_ids['lane_ids']))
        for i, lane in enumerate(start_lanes):
            candidate_route.extend(
                self.traverse_topology(lane, route_dict)
            )

        # generate the reference lanes
        candidate_reference_lanes = []
        for path in candidate_route:
            center_states_list = []
            for lane in path:
                center_states = lane['center_states']
                if center_states_list and center_states_list[-1] == center_states[0]:
                    # avoid repetation
                    center_states_list.extend(center_states[1:])
                else:
                    center_states_list.extend(center_states)
            candidate_reference_lanes.append(center_states_list)

        trimmed_paths, trimmed_path_length = [], []
        for center_states_list in candidate_reference_lanes:
            path, path_len = self._trim_discrete_path(current_state, center_states_list, max_length)
            trimmed_paths.append(path)
            trimmed_path_length.append(path_len)
        
        length_mask = np.array(trimmed_path_length) > 0.8 * max_length
        if length_mask.any() and not length_mask.all():
            trimmed_paths = [trimmed_paths[i] for i in np.where(length_mask)[0]]

        remove_index = set()
        for i in range(len(trimmed_paths)):
            for j in range(i + 1, len(trimmed_paths)):
                if j in remove_index:
                    continue
                min_len = min(len(trimmed_paths[i]), len(trimmed_paths[j]))
                diff = np.abs(
                    trimmed_paths[i][:min_len, :2] - trimmed_paths[j][:min_len, :2]
                ).sum(-1)
                if np.max(diff) < 0.5:
                    remove_index.add(j)

        reference_lanes = [
            trimmed_paths[i] for i in range(len(trimmed_paths)) if i not in remove_index
        ]

        return reference_lanes

    def get_lane_by_id(self, road_id: int, lane_id: int):
        lane = self.all_lane_object_gdf[(self.all_lane_object_gdf['road_id'] == road_id) & (self.all_lane_object_gdf['lane_id'] == lane_id)]
        if not lane.empty:
                lane_copy = lane.copy()
                lane_copy = lane_copy.reset_index(drop=True)

                def extract_value(cell):
                    if isinstance(cell, dict) and 0 in cell:
                        return cell[0]
                    else:
                        return cell

                lane_copy = lane_copy.applymap(extract_value)

                lane_dict = lane_copy.iloc[0].to_dict()
                return lane_dict
        else:
            return None

    def has_matching_pair(self, pairs_set, new_road_id, new_lane_id):
        for road_id, lane_id in pairs_set:
            if road_id == new_road_id and new_lane_id * lane_id > 0:
                return True
        return False

    def traverse_topology(self, current_lane, route_dict, search_depth=20):
        candidate_route = []
        def dfs_search(cur_lane, visited):
            if cur_lane is None:
                return
            visited.append(cur_lane)

            in_route_next_lane = [
                self.get_lane_by_id(next_road_id, next_lane_id)
                for next_road_id, next_lane_id in cur_lane['topology']
                if self.has_matching_pair(route_dict, next_road_id, next_lane_id)
                and not (next_road_id == cur_lane['road_id'] and next_lane_id == cur_lane['lane_id'])
            ]

            if (
                len(in_route_next_lane) == 0
                or len(visited) == search_depth
            ):
                candidate_route.append(visited)
                return

            for next_lane in in_route_next_lane:
                dfs_search(next_lane, visited.copy())

        dfs_search(current_lane, [])

        return candidate_route

    def _trim_discrete_path(
        self, current_state: CarlaAgentState, discrete_path: List[StateSE2], max_length=120
    ):
        if discrete_path is None:
            return None

        path = create_path_from_se2(discrete_path)
        linestring = path_to_linestring(discrete_path)

        start_progress = path.get_start_progress()
        end_progress = path.get_end_progress()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cur_progress = linestring.project(Point(*current_state.rear_axle.point.array))

        cut_start = max(start_progress, min(cur_progress, end_progress))
        cur_end = min(cur_progress + max_length, end_progress)

        trimmed_path = trim_path(path, cut_start, cur_end)
        path_length = cur_end - cut_start

        np_trimmed_path = np.array([[p.x, p.y, p.heading] for p in trimmed_path])

        return np_trimmed_path, path_length


class FakeCarlaWaypoint:
    def __init__(self, road_id, lane_id):
        self.road_id = road_id
        self.lane_id = lane_id


def test_query_proximal_map_data():
    map_api = CarlaMap(
        carla_town_name=args.carla_town,
    )
    x = -2135
    y = -6457
    ref_point = Point(x, y)
    radius = 300

    # proximal map objects
    proximal_map_data = map_api.query_proximal_map_data(ref_point, radius)

    # Visualize the results
    fig, ax = plt.subplots(figsize=(10, 10))

    # plot the query area
    buffer = ref_point.buffer(radius)
    gpd.GeoSeries(buffer).plot(ax=ax, color='grey', alpha=0.2, label='Query Area')

    ax.set_xlim(x - radius, x + radius)
    ax.set_ylim(y - radius, y + radius)
    
    # Plot Lanes
    if SemanticMapLayer.LANE in proximal_map_data:
        for lane in proximal_map_data[SemanticMapLayer.LANE]:
            if lane.geometry is not None:
                gpd.GeoSeries(lane.geometry).plot(ax=ax, color='grey', label='Lane', alpha=0.6)
    
    # Plot Lane Connectors
    if SemanticMapLayer.LANE_CONNECTOR in proximal_map_data:
        for lane_connector in proximal_map_data[SemanticMapLayer.LANE_CONNECTOR]:
            if lane_connector.geometry is not None:
                gpd.GeoSeries(lane_connector.geometry).plot(ax=ax, color='green', label='Lane Connector', alpha=0.6)
    
    # Plot Crosswalks
    if SemanticMapLayer.CROSSWALK in proximal_map_data:
        for crosswalk in proximal_map_data[SemanticMapLayer.CROSSWALK]:
            if crosswalk.geometry is not None:
                gpd.GeoSeries(crosswalk.geometry).plot(ax=ax, color='grey', label='Crosswalk', alpha=0.8)

    plt.title("Visualization of Proximal Map Data")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    output_filename = f'{args.carla_town} Map API visualization.png'
    output_path = os.path.join(args.output_dir, output_filename)

    # Ensure the directory exists
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    plt.savefig(output_path, dpi=300)

def test_query_ref_line():
    map_api = CarlaMap(
        carla_town_name=args.carla_town,
    )
    with open(args.example_routes, 'r') as json_file:
        example_routes = json.load(json_file)
    
    start_point = example_routes['start_point_xy']
    route_road_ids = {'road_ids': example_routes['right']['road_ids'],
                      'lane_ids': example_routes['right']['lane_ids']}
    global_route_waypoints = example_routes['right']['state']
    ego_waypoint = FakeCarlaWaypoint(example_routes["start_point_road_id"], example_routes["start_point_lane_id"])
    ref_lane_max_length = 80

    ref_point= Point(start_point[0], start_point[1])

    # reference lanes
    state = StateSE2(x=start_point[0], y=start_point[1], heading=0)
    ego_state = get_sample_ego_state(center=state)
    reference_lanes = map_api.query_reference_lines([ego_waypoint], ego_state, route_road_ids, max_length=ref_lane_max_length)
    
    # Visualize the results
    fig, ax = plt.subplots(figsize=(10, 10))

    # plot the global route waypoints
    for wp in global_route_waypoints[::5]:
        x = wp[0]
        y = wp[1]
        heading = wp[2]

        center_x = x + np.cos(heading) * 2
        center_y = y + np.sin(heading) * 2

        rectangle = patches.Rectangle(
            (center_x - 2, center_y - 2),
            width=4,
            height=4,
            color='yellow',
            ec='yellow',
            alpha=0.5
        )
        t = patches.transforms.Affine2D().rotate_deg_around(center_x, center_y, -np.degrees(heading)) + ax.transData

        rectangle.set_transform(t)
        ax.add_patch(rectangle)

    # plot the ref point
    gpd.GeoSeries(ref_point).plot(ax=ax, color='red', marker='*', markersize=100, label='Reference Point')

    # Plot reference lanes
    for path in reference_lanes:
        for point in path[::10]:
            x = point[0]
            y = point[1]
            heading = point[2]
            dx = np.cos(heading) * 3
            dy = np.sin(heading) * 3
            ax.arrow(x, y, dx, dy, head_width=1.2, head_length=1.5, fc='red', ec='red', alpha=0.7)

    plt.title("Visualization of Proximal Map Data")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    output_filename = f'{args.carla_town} Map API visualization.png'
    output_path = os.path.join(args.output_dir, output_filename)

    # Ensure the directory exists
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    plt.savefig(output_path, dpi=300)


def main():

    test_query_proximal_map_data()
    
    # test_query_ref_line()


if __name__ == '__main__':
    import json
    import argparse
    import matplotlib.patches as patches
    parser = argparse.ArgumentParser()
    parser.add_argument('--example_routes', default='tools/test/example_routes.json')
    parser.add_argument('--output_dir', default='data/map_data/anno')
    parser.add_argument('--carla_town', '-town', default='Town05')
    args = parser.parse_args()

    main()