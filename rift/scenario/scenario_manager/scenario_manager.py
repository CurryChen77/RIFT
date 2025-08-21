#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : scenario_manager.py
@Date    : 2023/10/4
"""
import time
import numpy as np
import py_trees
from typing import List
from datetime import datetime
from collections import defaultdict

import carla
from rift.cbv.planning.pluto.utils.nuplan_state_utils import CarlaAgentState
from rift.gym_carla.utils.common import convert_log_lat
from rift.scenario.scenario_manager.route_scenario import RouteScenario
from rift.scenario.tools.carla_data_provider import CarlaDataProvider
from rift.scenario.tools.metrics import compute_ego_critical_metrics
from rift.scenario.tools.route_scenario_configuration import RouteScenarioConfiguration
from rift.scenario.tools.timer import GameTime
from rift.scenario.statistics_manager import StatisticsManager
from rift.util.logger import Logger


def get_weather_id(weather_conditions):
    from xml.etree import ElementTree as ET
    tree = ET.parse('rift/scenario/route/weather.xml')
    root = tree.getroot()
    def conditions_match(weather, conditions):
        for (key, value) in weather:
            if key == 'route_percentage' : continue
            if str(getattr(conditions, key))!= value:
                return False
        return True
    for case in root.findall('case'):
        weather = case[0].items()
        if conditions_match(weather, weather_conditions):
            return case.items()[0][1]
    return None


class ScenarioManager(object):
    """
        Dynamic version scenario manager class. This class holds all functionality
        required to initialize, trigger, update and stop a scenario.
    """

    def __init__(self, env_params, statistics_manager: StatisticsManager, logger: Logger):
        self.env_params = env_params
        self.logger = logger
        self.running = False

        self.statistics_manager = statistics_manager

        self._reset()

    def _reset(self):
        # self.scenario = None
        self.map = CarlaDataProvider.get_map()
        self.route_scenario = None
        self.ego_vehicle = None
        self.ego_collision = False
        self.CBVs_data = defaultdict(lambda: {
            'speed': [],
            'acc': [],
            'jerk': [],
            'location': [],
        })

        self.running = False
        self._timestamp_last_run = 0.0

        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = 0.0
        self.start_game_time = 0.0
        self.end_system_time = 0.0
        self.end_game_time = 0.0

        self.running_record = []

        GameTime.restart()

    def clean_up(self):
        if self.route_scenario is not None:
            self.route_scenario.clean_up()

    def load_scenario(self, scenario, config: RouteScenarioConfiguration):
        self._reset()
        self.route_scenario: RouteScenario = scenario  # the RouteScenario

        if self.statistics_manager:
            self.prepare_statistic(config)
            self.statistics_manager.set_scenario(scenario)

        self.scenario_tree: py_trees.composites.Parallel = scenario.scenario_tree
        self.ego_vehicle = scenario.ego_vehicle

    def prepare_statistic(self,config: RouteScenarioConfiguration):

        route_name = f"{config.name}_rep{config.repetition_index}"
        scenario_name = 'None'
        town_name = str(config.town)
        weather_id = get_weather_id(config.weather[0][1])
        currentDateAndTime = datetime.now()
        currentTime = currentDateAndTime.strftime("%m_%d_%H_%M_%S")
        save_name = f"{route_name}_{town_name}_{scenario_name}_{weather_id}_{currentTime}"

        self.route_index = config.index  # the unique route_index of the route scenario
        self.statistics_manager.create_route_data(route_name, scenario_name, weather_id, save_name, town_name, config.index)

    def run_scenario(self):
        self.running = True
        self.start_system_time = time.time()
        self.start_game_time = GameTime.get_time()
        # generate the background vehicles
        self.route_scenario.initialize_actors()
    
    def stop_scenario(self):
        self.running = False

    def register_statistics(self, entry_status="Started", crash_message=""):
        """
        Computes and saves the route statistics
        """
        self.logger.log("\033[1m>> Registering the route statistics\033[0m", color='yellow')
        self.statistics_manager.save_entry_status(entry_status)
        self.statistics_manager.compute_route_statistics(
            self.route_index, self.scenario_duration_system, self.scenario_duration_game, crash_message
        )

    def update_running_status(self, CBVs_collision):
        # tick the scenario tree to update the running status
        self.scenario_tree.tick_once()

        # # for tree debugging
        # print("\n")
        # py_trees.display.print_ascii_tree(self.scenario_tree, show_status=True)
        # sys.stdout.flush()

        # update the status of the scenario manager
        if self.scenario_tree.status != py_trees.common.Status.RUNNING:
            self.running = False
        
        # update ego collision and CBV collision
        self.ego_collision = any(
            criterion.name == "Ego_collision" and criterion.status == py_trees.common.Status.FAILURE
            for criterion in self.route_scenario.get_criteria()
        )

        self.compute_duration_time()

        if self.statistics_manager:
            # Update live statistics
            self.statistics_manager.compute_route_statistics(
                self.route_index,  # the unique route index of the route scenario
                self.scenario_duration_system,
                self.scenario_duration_game,
                failure_message=""
            )
            CBVs_data, ego_data = self.compute_live_statistics(CBVs_collision)
            self.statistics_manager.write_live_results(
                self.route_index,  # the unique route index of the route scenario
                ego_data=ego_data,
                CBVs_data=CBVs_data,
            )

    def get_update(self, timestamp, ego_action, cbv_actions):
        if self._timestamp_last_run < timestamp.elapsed_seconds and self.running:
            self._timestamp_last_run = timestamp.elapsed_seconds

            # update ego action and cbv action
            self.route_scenario.update_actions(ego_action, cbv_actions)

            # update the background actors
            self.route_scenario.activate_background_actors() if len(self.route_scenario.unactivated_actors) > 0 else None


    def compute_duration_time(self):
        """
        Computes system and game duration times
        """
        self.end_system_time = time.time()
        self.end_game_time = GameTime.get_time()
        
        self.scenario_duration_system = self.end_system_time - self.start_system_time
        self.scenario_duration_game = self.end_game_time - self.start_game_time

    def compute_live_statistics(self, CBVs_collision):
        """
        Computes the scenario metrics
        """
        delta_time = CarlaDataProvider.get_tick_time()
        cur_CBVs = CarlaDataProvider.get_CBVs_by_ego(self.ego_vehicle.id)
        current_CBV_ids = set(cur_CBVs.keys())
        existing_CBV_ids = set(self.CBVs_data.keys())

        # Initialize new CBVs with empty data structures
        for cbv_id in current_CBV_ids - existing_CBV_ids:
            self.CBVs_data[cbv_id] = {
                'speed': [],
                'acc': [],
                'jerk': [],
                'location': [],
            }

        # Prepare metrics containers
        CBVs_total_game_time = 0.0
        CBVs_off_road_time = 0.0
        CBVs_uncomfortable_time = 0.0
        CBVs_total_progress = 0.0
        CBVs_collision_count = 0
        CBVs_target_speed = []
        CBVs_delta_speed = []

        # Process all current CBVs
        for CBV_id, CBV in cur_CBVs.items():
            cbv_data = self.CBVs_data[CBV_id]
            current_loc = CarlaDataProvider.get_location(CBV)
            current_state = CarlaDataProvider.get_current_state(CBV)
            current_acc = current_state.dynamic_car_state.rear_axle_acceleration_2d
            current_acc_mag = current_acc.magnitude()
            current_speed_mag = current_state.dynamic_car_state.rear_axle_velocity_2d.magnitude()

            # record CBVs game time
            CBVs_total_game_time += delta_time

            # Collect target speed of the CBVs vehicle setting
            target_speed = CBV.get_speed_limit() / 3.6  # Convert to m/s
            CBVs_target_speed.append(target_speed)
            CBVs_delta_speed.append(target_speed - current_speed_mag)
            
            # Update speed and acceleration
            cbv_data['speed'].append(current_speed_mag)
            cbv_data['acc'].append(current_acc_mag)
            
            # Calculate jerk (with safe handling of initial state)
            if len(cbv_data['acc']) >= 2:
                prev_acc = cbv_data['acc'][-2]
            else:
                prev_acc = 0.0
            current_jerk = (current_acc_mag - prev_acc) / delta_time if delta_time > 1e-6 else 0.0
            cbv_data['jerk'].append(current_jerk)
            
            # Calculate progress (distance from last location)
            if cbv_data['location']:  # Check if there is historical location data
                prev_loc = cbv_data['location'][-1]
                CBVs_total_progress += current_loc.distance(prev_loc)
            cbv_data['location'].append(current_loc)  # Append new location
            
            # Check off-road status
            if not self.map.get_waypoint(current_loc, project_to_road=False, lane_type=carla.LaneType.Driving | carla.LaneType.Parking):
                CBVs_off_road_time += delta_time

            # Check uncomfortable status
            if not((-4.05 < current_acc.x < 2.40) and (-4.89 < current_acc.y < 4.89) and (-8.37 < current_jerk < 8.37)):
                CBVs_uncomfortable_time += delta_time

            # Update collision count
            if CBVs_collision.get(CBV_id, False):
                CBVs_collision_count += 1

        # Remove old CBVs
        for cbv_id in existing_CBV_ids - current_CBV_ids:
            del self.CBVs_data[cbv_id]

        # Aggregate data for output
        CBVs_data = {
            'speed': [v for data in self.CBVs_data.values() for v in data['speed']],
            'acc': [v for data in self.CBVs_data.values() for v in data['acc']],
            'jerk': [v for data in self.CBVs_data.values() for v in data['jerk']],
            'total_game_time': CBVs_total_game_time,
            'off_road_game_time': CBVs_off_road_time,
            'uncomfortable_game_time': CBVs_uncomfortable_time,
            'target_speed': CBVs_target_speed,
            'delta_speed': CBVs_delta_speed,
            'total_progress': CBVs_total_progress,
            'collision_count': CBVs_collision_count,
            'new_cbv_count': len(current_CBV_ids - existing_CBV_ids),
            'reach_goal_count': len(CarlaDataProvider.get_CBVs_reach_goal_by_ego(self.ego_vehicle.id)),
        }
        
        ego_loc = CarlaDataProvider.get_location(self.ego_vehicle)
        ego_speed = CarlaDataProvider.get_velocity(self.ego_vehicle).length()

        # ego and nearby agent states
        ego_state = CarlaDataProvider.get_current_state(self.ego_vehicle)
        ego_nearby_agent_states: List[CarlaAgentState] = []
        for agent in CarlaDataProvider.get_ego_nearby_agents(self.ego_vehicle.id):
            ego_nearby_agent_states.append(CarlaDataProvider.get_current_state(agent))

        ego_data = {
            'speed': ego_speed,
            'control': py_trees.blackboard.Blackboard().get("Ego_action"),
            'location': ego_loc,
        }

        ego_critical_metrics = compute_ego_critical_metrics(ego_state, ego_nearby_agent_states)
        ego_data.update(ego_critical_metrics)

        return CBVs_data, ego_data
