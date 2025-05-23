#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : carla_data_provider.py
@Date    : 2023/10/4
"""

import math
import re
from collections import deque

from rift.scenario.tools.exception import SpawnRuntimeError
from nuplan_plugin.actor_state.tracked_objects_types import TrackedObjectType
from nuplan_plugin.actor_state.vehicle_parameters import VehicleParameters
from numpy import random
import time
import carla


class CarlaDataProvider(object): 
    """
        This module provides all frequently used data from CARLA via local buffers to avoid blocking calls to CARLA
    """

    _actor_velocity_map = {}
    _actor_acceleration_map = {}
    _actor_location_map = {}
    _actor_transform_map = {}
    _actor_history_state_map = {}
    _traffic_light_map = {}
    _carla_actor_pool = {}
    _all_actors = None
    _client = None
    _world = None
    _map = None
    _map_api = None
    _global_route_planner = None
    _gps_info = None
    _sync_flag = False
    _frame_rate = None
    _spawn_points = None
    _scenario_actors = {}
    _ego_nearby_agents = {}
    _CBV_nearby_agents = {}
    _spawn_index = 0
    _desired_speed = 8  # m/s
    _tick_time = 0.1
    _vehicle_info = {}
    _blueprint_library = None
    _ego_vehicle_route = {}
    _egos = {}
    _traffic_manager_port = 8000
    _traffic_random_seed = 0
    _rng = random.RandomState(_traffic_random_seed)

    @staticmethod
    def register_actor(actor):
        """
            Add new actor to dictionaries. If actor already exists, throw an exception
        """
        if actor in CarlaDataProvider._actor_velocity_map:
            raise KeyError("Vehicle '{}' already registered. Cannot register twice!".format(actor.id))
        else:
            CarlaDataProvider._actor_velocity_map[actor] = None

        if actor in CarlaDataProvider._actor_acceleration_map:
            raise KeyError("Vehicle '{}' already registered. Cannot register twice!".format(actor.id))
        else:
            CarlaDataProvider._actor_acceleration_map[actor] = None

        if actor in CarlaDataProvider._actor_location_map:
            raise KeyError("Vehicle '{}' already registered. Cannot register twice!".format(actor.id))
        else:
            CarlaDataProvider._actor_location_map[actor] = None

        if actor in CarlaDataProvider._actor_transform_map:
            raise KeyError("Vehicle '{}' already registered. Cannot register twice!".format(actor.id))
        else:
            CarlaDataProvider._actor_transform_map[actor] = None

        if actor in CarlaDataProvider._actor_history_state_map:
            raise KeyError("Vehicle '{}' already registered. Cannot register twice!".format(actor.id))
        else:
            CarlaDataProvider._vehicle_info[actor] = CarlaDataProvider.initialize_vehicle_infos(actor)
            CarlaDataProvider._actor_history_state_map[actor] = deque(maxlen=int(CarlaDataProvider._frame_rate * 2 + 5))

    @staticmethod
    def register_actors(actors):
        """
            Add new set of actors to dictionaries
        """
        for actor in actors:
            CarlaDataProvider.register_actor(actor)

    @staticmethod
    def get_vehicle_info(actor):
        if actor in CarlaDataProvider._vehicle_info:
            return CarlaDataProvider._vehicle_info[actor]
        
        # We are intentionally not throwing here
        print('{}.vehicle info: {} not found!' .format(__name__, actor))
        return None

    @staticmethod
    def on_carla_tick():
        """
            Callback from CARLA
        """
        for actor in CarlaDataProvider._actor_velocity_map:
            if actor is not None and actor.is_alive:
                CarlaDataProvider._actor_velocity_map[actor] = actor.get_velocity()

        for actor in CarlaDataProvider._actor_acceleration_map:
            if actor is not None and actor.is_alive:
                CarlaDataProvider._actor_acceleration_map[actor] = actor.get_acceleration()

        for actor in CarlaDataProvider._actor_location_map:
            if actor is not None and actor.is_alive:
                CarlaDataProvider._actor_location_map[actor] = actor.get_location()

        for actor in CarlaDataProvider._actor_transform_map:
            if actor is not None and actor.is_alive:
                CarlaDataProvider._actor_transform_map[actor] = actor.get_transform()            

        world = CarlaDataProvider._world
        if world is None:
            print(">> WARNING: CarlaDataProvider couldn't find the world")

        CarlaDataProvider._all_actors = None

    @staticmethod
    def get_velocity(actor):
        """
            returns the absolute velocity for the given actor
        """
        velocity = CarlaDataProvider._actor_velocity_map.get(actor, None)
        if velocity is None:
            print(f'{__name__}.get_velocity: {actor.id} not found!')
        return velocity

    @staticmethod
    def get_acceleration(actor):
        """
            returns the absolute acceleration for the given actor
        """
        acceleration = CarlaDataProvider._actor_acceleration_map.get(actor, None)
        if acceleration is None:
            print(f'{__name__}.get_acceleration: {actor.id} not found!')
        return acceleration

    @staticmethod
    def get_location(actor):
        """
            returns the location for the given actor
        """
        location = CarlaDataProvider._actor_location_map.get(actor, None)
        if location is None:
            print(f'{__name__}.get_location: {actor.id} not found!')
        return location

    @staticmethod
    def get_transform(actor):
        """
            returns the transform for the given actor
        """
        transform = CarlaDataProvider._actor_transform_map.get(actor, None)
        if transform is None:
            print(f'{__name__}.get_transform: {actor.id} not found!')
        return transform

    @staticmethod
    def get_all_actors():
        """
        @return all the world actors. This is an expensive call, hence why it is part of the CDP,
        but as this might not be used by everyone, only get the actors the first time someone
        calls asks for them. 'CarlaDataProvider._all_actors' is reset each tick to None.
        """
        if CarlaDataProvider._all_actors:
            return CarlaDataProvider._all_actors

        CarlaDataProvider._all_actors = CarlaDataProvider._world.get_actors()
        return CarlaDataProvider._all_actors

    @staticmethod
    def add_history_state(actor, actor_state):
        if actor is not None and actor.is_alive and actor in CarlaDataProvider._actor_history_state_map.keys():
            CarlaDataProvider._actor_history_state_map[actor].append(actor_state)
        else:
            print(f"can't add history_state to actor {actor.id}")

    @staticmethod
    def get_history_state(actor):
        """
            returns the history state for the given actor
        """
        state = CarlaDataProvider._actor_history_state_map.get(actor, None)
        if state is None:
            print(f"{__name__}.get_history_state: {actor.id} not found!")
        elif len(state) == 0:
            print(f"{__name__}.get_history_state: {actor.id} get empty deque!")
        return state

    @staticmethod
    def get_current_state(actor):
        """
            returns the state for the given actor
        """
        state = CarlaDataProvider._actor_history_state_map.get(actor, None)
        if state is None:
            print(f"{__name__}.get_current_state: {actor.id} not found!")
            current_state = None
        elif len(state) == 0:
            print(f"{__name__}.get_history_state: {actor.id} get empty deque!")
            current_state = None
        else:
            current_state = state[-1]
        return current_state

    @staticmethod
    def set_tick_time(tick_time):
        """
            Set the tick time
        """
        CarlaDataProvider._tick_time = tick_time

    @staticmethod
    def get_tick_time():
        """
            Get the tick time
        """
        return CarlaDataProvider._tick_time

    @staticmethod
    def set_desired_speed(desired_speed):
        """
            Set the Ego min distance across nearby vehicles
        """
        CarlaDataProvider._desired_speed = desired_speed

    @staticmethod
    def get_desired_speed():
        """
            Get the Ego min distance across nearby vehicles
        """
        return CarlaDataProvider._desired_speed

    @staticmethod
    def set_client(client):
        """
            Set the CARLA client
        """
        CarlaDataProvider._client = client

    @staticmethod
    def get_client():
        """
            Get the CARLA client
        """
        return CarlaDataProvider._client

    @staticmethod
    def set_world(world, town):
        """
            Set the world and world settings
        """
        CarlaDataProvider._world = world
        CarlaDataProvider._sync_flag = world.get_settings().synchronous_mode
        CarlaDataProvider._map = world.get_map()
        CarlaDataProvider._blueprint_library = world.get_blueprint_library()
        CarlaDataProvider.reset_spawn_points()
        CarlaDataProvider.prepare_map()

    @staticmethod
    def get_world():
        """
            Return world
        """
        return CarlaDataProvider._world

    @staticmethod
    def set_map_api(map_api):
        """
            Set current town map api
        """
        CarlaDataProvider._map_api = map_api

    @staticmethod
    def get_map_api():
        """
            Return Current town map api
        """
        return CarlaDataProvider._map_api

    @staticmethod
    def set_gps_info(gps_info):
        """
            Set gps info
        """
        CarlaDataProvider._gps_info = gps_info

    @staticmethod
    def get_gps_info():
        """
            Return gps info
        """
        return CarlaDataProvider._gps_info

    @staticmethod
    def set_global_route_planner(grp):
        """
            Set current global route planner
        """
        CarlaDataProvider._global_route_planner = grp

    @staticmethod
    def get_global_route_planner():
        """
            Return Current global_route_planner
        """
        return CarlaDataProvider._global_route_planner

    @staticmethod
    def get_map(world=None):
        """
            Get the current map
        """
        if CarlaDataProvider._map is None:
            if world is None:
                if CarlaDataProvider._world is None:
                    raise ValueError("class member \'world'\' not initialized yet")
                else:
                    CarlaDataProvider._map = CarlaDataProvider._world.get_map()
            else:
                CarlaDataProvider._map = world.get_map()

        return CarlaDataProvider._map

    @staticmethod
    def get_traffic_random_seed():
        """
            return traffic random seed
        """
        return CarlaDataProvider._traffic_random_seed

    @staticmethod
    def set_traffic_random_seed(traffic_random_seed):
        """
            set traffic random seed
        """
        CarlaDataProvider._traffic_random_seed = traffic_random_seed
        CarlaDataProvider._rng = random.RandomState(traffic_random_seed)

    @staticmethod
    def get_frame_rate():
        """
            return frame rate
        """
        return CarlaDataProvider._frame_rate

    @staticmethod
    def set_frame_rate(frame_rate):
        """
            set frame rate
        """
        CarlaDataProvider._frame_rate = frame_rate

    @staticmethod
    def is_sync_mode():
        """
            return true if syncronuous mode is used
        """
        return CarlaDataProvider._sync_flag

    @staticmethod
    def find_weather_presets():
        """
            Get weather presets from CARLA
        """
        rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
        name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
        presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
        return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

    @staticmethod
    def prepare_map():
        """
            This function set the current map and loads all traffic lights for this map to _traffic_light_map
        """
        if CarlaDataProvider._map is None:
            CarlaDataProvider._map = CarlaDataProvider._world.get_map()

        # Parse all traffic lights
        CarlaDataProvider._traffic_light_map.clear()
        for traffic_light in CarlaDataProvider._world.get_actors().filter('*traffic_light*'):
            if traffic_light not in CarlaDataProvider._traffic_light_map.keys():
                CarlaDataProvider._traffic_light_map[traffic_light] = traffic_light.get_transform()
            else:
                raise KeyError("Traffic light '{}' already registered. Cannot register twice!".format(traffic_light.id))

    @staticmethod
    def annotate_trafficlight_in_group(traffic_light):
        """
            Get dictionary with traffic light group info for a given traffic light
        """
        dict_annotations = {'ref': [], 'opposite': [], 'left': [], 'right': []}

        # Get the waypoints
        ref_location = CarlaDataProvider.get_trafficlight_trigger_location(traffic_light)
        ref_waypoint = CarlaDataProvider.get_map().get_waypoint(ref_location)
        ref_yaw = ref_waypoint.transform.rotation.yaw

        group_tl = traffic_light.get_group_traffic_lights()
        for target_tl in group_tl:
            if traffic_light.id == target_tl.id:
                dict_annotations['ref'].append(target_tl)
            else:
                # Get the angle between yaws
                target_location = CarlaDataProvider.get_trafficlight_trigger_location(target_tl)
                target_waypoint = CarlaDataProvider.get_map().get_waypoint(target_location)
                target_yaw = target_waypoint.transform.rotation.yaw

                diff = (target_yaw - ref_yaw) % 360
                if diff > 330:
                    continue
                elif diff > 225:
                    dict_annotations['right'].append(target_tl)
                elif diff > 135.0:
                    dict_annotations['opposite'].append(target_tl)
                elif diff > 30:
                    dict_annotations['left'].append(target_tl)

        return dict_annotations

    @staticmethod
    def get_trafficlight_trigger_location(traffic_light):    # pylint: disable=invalid-name
        """
            Calculates the yaw of the waypoint that represents the trigger volume of the traffic light
        """
        def rotate_point(point, angle):
            """
                rotate a given point by a given angle
            """
            x_ = math.cos(math.radians(angle)) * point.x - math.sin(math.radians(angle)) * point.y
            y_ = math.sin(math.radians(angle)) * point.x - math.cos(math.radians(angle)) * point.y
            return carla.Vector3D(x_, y_, point.z)

        base_transform = traffic_light.get_transform()
        base_rot = base_transform.rotation.yaw
        area_loc = base_transform.transform(traffic_light.trigger_volume.location)
        area_ext = traffic_light.trigger_volume.extent

        point = rotate_point(carla.Vector3D(0, 0, area_ext.z), base_rot)
        point_location = area_loc + carla.Location(x=point.x, y=point.y)
        return carla.Location(point_location.x, point_location.y, point_location.z)

    @staticmethod
    def update_light_states(ego_light, annotations, states, freeze=False, timeout=1000000000):
        """
            Update traffic light states
        """
        reset_params = []

        for state in states:
            relevant_lights = []
            if state == 'ego':
                relevant_lights = [ego_light]
            else:
                relevant_lights = annotations[state]
            for light in relevant_lights:
                prev_state = light.get_state()
                prev_green_time = light.get_green_time()
                prev_red_time = light.get_red_time()
                prev_yellow_time = light.get_yellow_time()
                reset_params.append({
                    'light': light, 
                    'state': prev_state, 
                    'green_time': prev_green_time, 
                    'red_time': prev_red_time, 
                    'yellow_time': prev_yellow_time
                })
                light.set_state(states[state])
                if freeze:
                    light.set_green_time(timeout)
                    light.set_red_time(timeout)
                    light.set_yellow_time(timeout)

        return reset_params

    @staticmethod
    def reset_lights(reset_params):
        """
            Reset traffic lights
        """
        for param in reset_params:
            param['light'].set_state(param['state'])
            param['light'].set_green_time(param['green_time'])
            param['light'].set_red_time(param['red_time'])
            param['light'].set_yellow_time(param['yellow_time'])

    @staticmethod
    def get_next_intersection_location(location):
        waypoint = CarlaDataProvider.get_map().get_waypoint(location)
        # Create a list of all waypoints until the next intersection
        list_of_waypoints = []
        while waypoint and not waypoint.is_intersection:
            list_of_waypoints.append(waypoint)
            waypoint = waypoint.next(2.0)[0]
        # move one more step ahead in the intersection
        waypoint = waypoint.next(10.0)[0]
        list_of_waypoints.append(waypoint)
        next_intersection_location = list_of_waypoints[-1].transform.location if list_of_waypoints else location
        # CarlaDataProvider.get_world().debug.draw_point(next_intersection_location + carla.Location(z=2.0), size=0.3, color=carla.Color(0, 0, 255, 0), life_time=-1)
        return next_intersection_location

    @staticmethod
    def get_next_traffic_light(actor, use_cached_location=True, use_transform=False):
        """
            returns the next relevant traffic light for the provided actor
        """

        if use_transform:
            location = actor.location
        else:
            if not use_cached_location:
                location = actor.get_transform().location
            else:
                location = CarlaDataProvider.get_location(actor)

        waypoint = CarlaDataProvider.get_map().get_waypoint(location)
        # Create list of all waypoints until next intersection
        list_of_waypoints = []
        while waypoint and not waypoint.is_intersection:
            list_of_waypoints.append(waypoint)
            waypoint = waypoint.next(2.0)[0]

        # If the list is empty, the actor is in an intersection
        if not list_of_waypoints:
            return None

        relevant_traffic_light = None
        distance_to_relevant_traffic_light = float("inf")

        for traffic_light in CarlaDataProvider._traffic_light_map:
            if hasattr(traffic_light, 'trigger_volume'):
                tl_t = CarlaDataProvider._traffic_light_map[traffic_light]
                transformed_tv = tl_t.transform(traffic_light.trigger_volume.location)

                distance = carla.Location(transformed_tv).distance(list_of_waypoints[-1].transform.location)

                if distance < distance_to_relevant_traffic_light:
                    relevant_traffic_light = traffic_light
                    distance_to_relevant_traffic_light = distance

        return relevant_traffic_light

    @staticmethod
    def set_all_traffic_light(traffic_light_state, timeout):
        for traffic_light in CarlaDataProvider._traffic_light_map:
            if hasattr(traffic_light, 'trigger_volume'):
                traffic_light.set_state(traffic_light_state)
                traffic_light.set_green_time(timeout)
                traffic_light.freeze(True)  # freeze the traffic light as green
            else:
                print("got traffic light without trigger volume")

    @staticmethod
    def set_ego_vehicle_route(ego, route, env_id):
        """
            Set the route of the ego vehicle
        """
        CarlaDataProvider._ego_vehicle_route[ego.id] = route
        CarlaDataProvider._egos[env_id] = ego
        CarlaDataProvider._scenario_actors[ego.id] = {'BVs': {}, 'CBVs': {}, 'CBVs_reach_goal': {}}  # create an empty dictionary for each scenario to record all BVs
        CarlaDataProvider._CBV_nearby_agents[ego.id] = {}  # create an empty dictionary for each scenario to record all CBVs nearby agents

    @staticmethod
    def set_scenario_actors(ego, actors):
        for actor in actors:
            CarlaDataProvider._scenario_actors[ego.id]['BVs'][actor.id] = actor

    @staticmethod
    def get_scenario_actors():
        return CarlaDataProvider._scenario_actors

    @staticmethod
    def get_scenario_actors_by_ego(ego_id):
        return CarlaDataProvider._scenario_actors[ego_id]

    @staticmethod
    def get_CBVs_by_ego(ego_id):
        return CarlaDataProvider._scenario_actors[ego_id]['CBVs']

    @staticmethod
    def get_CBVs_reach_goal_by_ego(ego_id):
        return CarlaDataProvider._scenario_actors[ego_id]['CBVs_reach_goal']

    @staticmethod
    def add_CBV(ego, actor):
        # the selected CBV may not belong to the original ego, so we need to loop through all ego actors to pop CBV
        for ego_data in CarlaDataProvider._scenario_actors.values():
            if ego_data['BVs'].pop(actor.id, None) is not None:
                break
        CarlaDataProvider._scenario_actors[ego.id]['CBVs'][actor.id] = actor

    @staticmethod
    def CBV_reach_goal(ego, actor):
        CarlaDataProvider._scenario_actors[ego.id]['CBVs_reach_goal'][actor.id] = actor

    @staticmethod
    def CBV_back_to_BV(ego, actor):
        CarlaDataProvider._scenario_actors[ego.id]['CBVs'].pop(actor.id, None)
        CarlaDataProvider._scenario_actors[ego.id]['BVs'][actor.id] = actor

    @staticmethod
    def CBV_terminate(ego, actor):
        CarlaDataProvider._scenario_actors[ego.id]['CBVs'].pop(actor.id, None)
    
    @staticmethod
    def get_ego_vehicle_by_env_id(env_id):
        return CarlaDataProvider._egos[env_id]

    @staticmethod
    def get_all_ego_vehicles():
        """
            Set the route of the ego vehicle
        """
        return CarlaDataProvider._egos.values()

    @staticmethod
    def get_first_ego_transform():
        """
            Set the first ego location (if it is not terminated)
        """
        return CarlaDataProvider.get_transform(CarlaDataProvider._egos[0])

    @staticmethod
    def set_ego_nearby_agents(ego, agents):
        """
            Set the ego_nearby_agents
        """
        CarlaDataProvider._ego_nearby_agents[ego.id] = agents

    @staticmethod
    def get_ego_nearby_agents(ego_id):
        """
            Get the ego_nearby_agents
        """
        return CarlaDataProvider._ego_nearby_agents[ego_id]
    
    @staticmethod
    def set_CBV_nearby_agents(ego, CBV, CBV_nearby_agents):
        """
            Set the CBV_nearby_agents
        """
        CarlaDataProvider._CBV_nearby_agents[ego.id][CBV.id] = CBV_nearby_agents

    @staticmethod
    def pop_CBV_nearby_agents(ego, CBV):
        """
            Remove the CBV_nearby_agents
        """
        CarlaDataProvider._CBV_nearby_agents[ego.id].pop(CBV.id)

    @staticmethod
    def get_CBV_nearby_agents(ego_id, CBV_id):
        """
            Get the CBV_nearby_agents
        """
        return CarlaDataProvider._CBV_nearby_agents[ego_id][CBV_id]

    @staticmethod
    def reset_spawn_points():
        """
            Generate spawn points for the current map
        """
        spawn_points = list(CarlaDataProvider.get_map(CarlaDataProvider._world).get_spawn_points())
        CarlaDataProvider._rng.shuffle(spawn_points)
        CarlaDataProvider._spawn_points = spawn_points
        CarlaDataProvider._spawn_index = 0

    @staticmethod
    def initialize_vehicle_infos(vehicle):
        vehicle_attributes = vehicle.attributes
        number_of_wheels = int(vehicle_attributes.get('number_of_wheels'))
        vehicle_extent = vehicle.bounding_box.extent

        # Calculate actual vehicle dimensions
        vehicle_length = vehicle_extent.x * 2.0  # Actual length
        vehicle_width = vehicle_extent.y * 2.0  # Actual width
        vehicle_height = vehicle_extent.z * 2.0  # Actual height

        # Reference dimensions
        reference_length = 4.049 + 1.127  # Total reference length (front + rear)
        reference_width = 2.297  # Reference width

        # Calculate average scaling ratio
        ratio = sum([
            vehicle_width / reference_width,
            vehicle_length / reference_length,
        ]) / 2.0

        vehicle_parameters = VehicleParameters(
            vehicle_name=vehicle.type_id,
            vehicle_type=vehicle_attributes.get('role_name', 'background'),
            width=reference_width * ratio,
            front_length=4.049 * ratio,
            rear_length=1.127 * ratio,
            wheel_base=3.089 * ratio,
            cog_position_from_rear_axle=1.67 * ratio,
            height=vehicle_height,
        )
        if number_of_wheels == 2:
            vehicle_type = TrackedObjectType.BICYCLE
        else:
            vehicle_type = TrackedObjectType.VEHICLE

        return vehicle_parameters, vehicle_type

    @staticmethod
    def create_blueprint(model, rolename='scenario', color=None, actor_category="car", safe=False):
        """
            Function to setup the blueprint of an actor given its model and other relevant parameters
        """

        _actor_blueprint_categories = {
            'car': 'vehicle.tesla.model3',
            'van': 'vehicle.volkswagen.t2',
            'truck': 'vehicle.carlamotors.carlacola',
            'trailer': '',
            'semitrailer': '',
            'bus': 'vehicle.volkswagen.t2',
            'motorbike': 'vehicle.kawasaki.ninja',
            'bicycle': 'vehicle.diamondback.century',
            'train': '',
            'tram': '',
            'pedestrian': 'walker.pedestrian.0001',
        }

        # Set the model
        try:
            blueprints = CarlaDataProvider._blueprint_library.filter(model)
            blueprints_ = []
            if safe:
                for bp in blueprints:
                    if bp.id.endswith('firetruck') or bp.id.endswith('ambulance') or int(bp.get_attribute('number_of_wheels')) != 4:
                        # Two wheeled vehicles take much longer to render + bicicles shouldn't behave like cars
                        continue
                    blueprints_.append(bp)
            else:
                blueprints_ = blueprints

            blueprint = CarlaDataProvider._rng.choice(blueprints_)
        except ValueError:
            # The model is not part of the blueprint library. Let's take a default one for the given category
            bp_filter = "vehicle.*"
            new_model = _actor_blueprint_categories[actor_category]
            if new_model != '':
                bp_filter = new_model
            print(">> WARNING: Actor model {} not available. Using instead {}".format(model, new_model))
            blueprint = CarlaDataProvider._rng.choice(CarlaDataProvider._blueprint_library.filter(bp_filter))

        # Set the color
        if color:
            if not blueprint.has_attribute('color'):
                print(">> WARNING: Cannot set Color ({}) for actor {} due to missing blueprint attribute".format(color, blueprint.id))
            else:
                default_color_rgba = blueprint.get_attribute('color').as_color()
                default_color = '({}, {}, {})'.format(default_color_rgba.r, default_color_rgba.g, default_color_rgba.b)
                try:
                    blueprint.set_attribute('color', color)
                except ValueError:
                    # Color can't be set for this vehicle
                    print(">> WARNING: Color ({}) cannot be set for actor {}. Using instead: ({})".format(color, blueprint.id, default_color))
                    blueprint.set_attribute('color', default_color)
        else:
            if blueprint.has_attribute('color') and rolename != 'hero':
                color = CarlaDataProvider._rng.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)

        # Make pedestrians mortal
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'false')

        # Set the rolename
        if blueprint.has_attribute('role_name'):
            blueprint.set_attribute('role_name', rolename)

        return blueprint

    @staticmethod
    def handle_actor_batch(batch, tick=True):
        """
            Forward a CARLA command batch to spawn actors to CARLA, and gather the responses.
            Returns list of actors on success, none otherwise
        """
        sync_mode = CarlaDataProvider.is_sync_mode()
        actors = []

        if CarlaDataProvider._client:
            responses = CarlaDataProvider._client.apply_batch_sync(batch, sync_mode and tick)
        else:
            raise ValueError("class member \'client'\' not initialized yet")

        # Wait (or not) for the actors to be spawned properly before we do anything
        if not tick:
            pass
        elif sync_mode:
            CarlaDataProvider._world.tick()
        else:
            CarlaDataProvider._world.wait_for_tick()

        actor_ids = [r.actor_id for r in responses if not r.error]
        for r in responses:
            if r.error:
                print(">> WARNING: Not all actors were spawned")
                break
        actors = list(CarlaDataProvider._world.get_actors(actor_ids))
        return actors

    @staticmethod
    def request_new_actor(
        model, 
        spawn_point, 
        rolename='scenario', 
        autopilot=False,
        random_location=False, 
        color=None, 
        actor_category="car",
        safe_blueprint=False, 
        tick=True,
        max_try=2
    ):
        """
            This method tries to create a new actor, returning it if successful (None otherwise).
        """
        blueprint = CarlaDataProvider.create_blueprint(model, rolename, color, actor_category, safe_blueprint)

        if random_location:
            actor = None
            max_attempts = 0
            while not actor and max_attempts < max_try:
                spawn_point = CarlaDataProvider._rng.choice(CarlaDataProvider._spawn_points)
                actor = CarlaDataProvider._world.try_spawn_actor(blueprint, spawn_point)
                max_attempts += 1
        else:
            # slightly lift the actor to avoid collisions with ground when spawning the actor
            # DO NOT USE spawn_point directly, as this will modify spawn_point permanently
            _spawn_point = carla.Transform(carla.Location(), spawn_point.rotation)
            _spawn_point.location.x = spawn_point.location.x
            _spawn_point.location.y = spawn_point.location.y
            _spawn_point.location.z = spawn_point.location.z
            # if cannot spawn actor, then change the x position of the actor
            max_attempts = 0
            actor = None
            while not actor and max_attempts < max_try:
                actor = CarlaDataProvider._world.try_spawn_actor(blueprint, _spawn_point)
                if not actor:
                    _spawn_point.location.z += 0.2
                    max_attempts += 1
        
        if not actor:
            raise SpawnRuntimeError(f"Failed to spawn actor {model} at {spawn_point.location} after {max_try} attempts")

        # De/activate the autopilot of the actor if it belongs to vehicle
        if autopilot:
            if actor.type_id.startswith('vehicle.'):
                actor.set_autopilot(autopilot, CarlaDataProvider._traffic_manager_port)
            else:
                print(">> WARNING: Tried to set the autopilot of a non vehicle actor")

        # Wait for the actor to be spawned properly before we do anything
        if not tick:
            pass
        elif CarlaDataProvider.is_sync_mode():
            CarlaDataProvider._world.tick()
        else:
            CarlaDataProvider._world.wait_for_tick()
        CarlaDataProvider._carla_actor_pool[actor.id] = actor
        CarlaDataProvider.register_actor(actor)

        return actor

    @staticmethod
    def request_new_actors(actor_list, safe_blueprint=False, tick=True):
        """
            This method tries to series of actor in batch. If this was successful, the new actors are returned, None otherwise.
                actor_list: list of ActorConfigurationData
        """
        SpawnActor = carla.command.SpawnActor                     
        PhysicsCommand = carla.command.SetSimulatePhysics         
        FutureActor = carla.command.FutureActor                   
        ApplyTransform = carla.command.ApplyTransform              
        SetAutopilot = carla.command.SetAutopilot                 
        SetVehicleLightState = carla.command.SetVehicleLightState  

        batch = []
        for actor in actor_list:
            # Get the blueprint
            blueprint = CarlaDataProvider.create_blueprint(actor.model, actor.rolename, actor.color, actor.category, safe_blueprint)

            # Get the spawn point
            transform = actor.transform
            if actor.random_location:
                if CarlaDataProvider._spawn_index >= len(CarlaDataProvider._spawn_points):
                    print("No more spawn points to use")
                    break
                else:
                    _spawn_point = CarlaDataProvider._spawn_points[CarlaDataProvider._spawn_index]  
                    CarlaDataProvider._spawn_index += 1
            else:
                _spawn_point = carla.Transform()
                _spawn_point.rotation = transform.rotation
                _spawn_point.location.x = transform.location.x
                _spawn_point.location.y = transform.location.y
                if blueprint.has_tag('walker'):
                    # On imported OpenDRIVE maps, spawning of pedestrians can fail.
                    # By increasing the z-value the chances of success are increased.
                    map_name = CarlaDataProvider._map.name.split("/")[-1]
                    if not map_name.startswith('OpenDrive'):
                        _spawn_point.location.z = transform.location.z + 0.2
                    else:
                        _spawn_point.location.z = transform.location.z + 0.8
                else:
                    _spawn_point.location.z = transform.location.z + 0.2

            # Get the command
            command = SpawnActor(blueprint, _spawn_point)
            command.then(SetAutopilot(FutureActor, actor.autopilot, CarlaDataProvider._traffic_manager_port))

            if actor.args is not None and 'physics' in actor.args and actor.args['physics'] == "off":
                command.then(ApplyTransform(FutureActor, _spawn_point)).then(PhysicsCommand(FutureActor, False))
            elif actor.category == 'misc':
                command.then(PhysicsCommand(FutureActor, True))
            if actor.args is not None and 'lights' in actor.args and actor.args['lights'] == "on":
                command.then(SetVehicleLightState(FutureActor, carla.VehicleLightState.All))

            batch.append(command)

        actors = CarlaDataProvider.handle_actor_batch(batch, tick)
        for actor in actors:
            if actor is None:
                continue
            CarlaDataProvider._carla_actor_pool[actor.id] = actor
            CarlaDataProvider.register_actor(actor)

        return actors

    @staticmethod
    def request_new_batch_actors(
        model, 
        amount, 
        spawn_points, 
        autopilot=False,
        random_location=False, 
        rolename='scenario',
        safe_blueprint=False, 
        tick=True
    ):
        """
            Simplified version of "request_new_actors". This method also create several actors in batch.
            Instead of needing a list of ActorConfigurationData, an "amount" parameter is used.
            This makes actor spawning easier but reduces the amount of configurability.
            Some parameters are the same for all actors (rolename, autopilot and random location) while others are randomized (color)
        """

        SpawnActor = carla.command.SpawnActor      
        SetAutopilot = carla.command.SetAutopilot 
        FutureActor = carla.command.FutureActor   

        batch = []
        for i in range(amount):
            # Get vehicle by model
            blueprint = CarlaDataProvider.create_blueprint(model, rolename, safe=safe_blueprint)

            if random_location:
                if CarlaDataProvider._spawn_index >= len(CarlaDataProvider._spawn_points):
                    print("No more spawn points to use. Spawned {} actors out of {}".format(i + 1, amount))
                    break
                else:
                    spawn_point = CarlaDataProvider._spawn_points[CarlaDataProvider._spawn_index]  
                    CarlaDataProvider._spawn_index += 1
            else:
                try:
                    spawn_point = spawn_points[i]
                except IndexError:
                    print("The amount of spawn points is lower than the amount of vehicles spawned")
                    break

            if spawn_point:
                batch.append(SpawnActor(blueprint, spawn_point).then(SetAutopilot(FutureActor, autopilot, CarlaDataProvider._traffic_manager_port)))

        time.sleep(1)  # need to sleep for a while for the batch_sync operation

        actors = CarlaDataProvider.handle_actor_batch(batch, tick)
        for actor in actors:
            if actor is None:
                continue
            CarlaDataProvider._carla_actor_pool[actor.id] = actor
            CarlaDataProvider.register_actor(actor)
        return actors

    @staticmethod
    def get_actors():
        """
            Return list of actors and their ids
            Note: iteritems from six is used to allow compatibility with Python 2 and 3
        """
        return CarlaDataProvider._carla_actor_pool

    @staticmethod
    def actor_id_exists(actor_id):
        """
            Check if a certain id is still at the simulation
        """
        if actor_id in CarlaDataProvider._carla_actor_pool:
            return True

        return False

    @staticmethod
    def get_hero_actor():
        """
            Get the actor object of the hero actor if it exists, returns none otherwise.
        """
        for actor_id in CarlaDataProvider._carla_actor_pool:
            if CarlaDataProvider._carla_actor_pool[actor_id].attributes['role_name'] == 'hero':
                return CarlaDataProvider._carla_actor_pool[actor_id]
        return None

    @staticmethod
    def get_actor_by_id(actor_id):
        """
            Get an actor from the pool by using its ID. If the actor does not exist, None is returned.
        """
        if actor_id in CarlaDataProvider._carla_actor_pool:
            return CarlaDataProvider._carla_actor_pool[actor_id]
        print("Non-existing actor id {}".format(actor_id))
        return None

    @staticmethod
    def remove_actor_by_id(actor_id):
        """
            Remove an actor from the pool using its ID
        """
        # use carla.command.DestroyActor instead of actor.destroy()
        DestroyActor = carla.command.DestroyActor
        if actor_id in CarlaDataProvider._carla_actor_pool:
            CarlaDataProvider._client.apply_batch([DestroyActor(CarlaDataProvider._carla_actor_pool[actor_id])])
            CarlaDataProvider._carla_actor_pool[actor_id] = None
            CarlaDataProvider._carla_actor_pool.pop(actor_id)
        else:
            print("Trying to remove a non-existing actor id {}".format(actor_id))

    @staticmethod
    def remove_actors_in_surrounding(location, distance):
        """
            Remove all actors from the pool that are closer than distance to the provided location
        """
        DestroyActor = carla.command.DestroyActor
        batch = []
        for actor_id in CarlaDataProvider._carla_actor_pool.copy():
            if CarlaDataProvider._carla_actor_pool[actor_id].get_location().distance(location) < distance:
                batch.append(DestroyActor(CarlaDataProvider._carla_actor_pool[actor_id]))
                CarlaDataProvider._carla_actor_pool.pop(actor_id)

        if CarlaDataProvider._client:
            try:
                CarlaDataProvider._client.apply_batch_sync(batch)
            except RuntimeError as e:
                if "time-out" in str(e):
                    pass
                else:
                    raise e

        # Remove all keys with None values
        CarlaDataProvider._carla_actor_pool = dict({k: v for k, v in CarlaDataProvider._carla_actor_pool.items() if v})

    @staticmethod
    def get_traffic_manager_port():
        """
            Get the port of the traffic manager.
        """
        return CarlaDataProvider._traffic_manager_port

    @staticmethod
    def set_traffic_manager_port(tm_port):
        """
            Set the port to use for the traffic manager.
        """
        CarlaDataProvider._traffic_manager_port = tm_port

    @staticmethod
    def clean_up_after_episode():
        """
            Cleanup and remove all entries from all dictionaries
        """
        DestroyActor = carla.command.DestroyActor       # pylint: disable=invalid-name
        batch = []

        for actor_id in CarlaDataProvider._carla_actor_pool.copy():
            actor = CarlaDataProvider._carla_actor_pool[actor_id]
            if actor is not None and actor.is_alive:
                batch.append(DestroyActor(actor))

        if CarlaDataProvider._client:
            try:
                CarlaDataProvider._client.apply_batch_sync(batch)
            except RuntimeError as e:
                if "time-out" in str(e):
                    pass
                else:
                    raise e

        CarlaDataProvider._actor_velocity_map.clear()
        CarlaDataProvider._actor_location_map.clear()
        CarlaDataProvider._actor_acceleration_map.clear()
        CarlaDataProvider._actor_transform_map.clear()
        CarlaDataProvider._actor_history_state_map.clear()
        CarlaDataProvider._ego_vehicle_route.clear()
        CarlaDataProvider._egos = {}
        CarlaDataProvider._carla_actor_pool = {}
        CarlaDataProvider._scenario_actors = {}
        CarlaDataProvider._ego_nearby_agents = {}
        CarlaDataProvider._CBV_nearby_agents = {}
        CarlaDataProvider._vehicle_info = {}
        CarlaDataProvider._spawn_points = None
        CarlaDataProvider._spawn_index = 0

    @staticmethod
    def clean_up():
        """
            Cleanup and remove all entries from all dictionaries
        """
        DestroyActor = carla.command.DestroyActor       # pylint: disable=invalid-name
        batch = []

        for actor_id in CarlaDataProvider._carla_actor_pool.copy():
            actor = CarlaDataProvider._carla_actor_pool[actor_id]
            if actor is not None and actor.is_alive:
                batch.append(DestroyActor(actor))

        if CarlaDataProvider._client:
            try:
                CarlaDataProvider._client.apply_batch_sync(batch)
            except RuntimeError as e:
                if "time-out" in str(e):
                    pass
                else:
                    raise e

        CarlaDataProvider._actor_velocity_map.clear()
        CarlaDataProvider._actor_location_map.clear()
        CarlaDataProvider._actor_acceleration_map.clear()
        CarlaDataProvider._actor_transform_map.clear()
        CarlaDataProvider._actor_history_state_map.clear()
        CarlaDataProvider._traffic_light_map.clear()
        CarlaDataProvider._map = None
        CarlaDataProvider._world = None
        CarlaDataProvider._map_api = None
        CarlaDataProvider._gps_info = None
        CarlaDataProvider._global_route_planner = None
        CarlaDataProvider._sync_flag = False
        CarlaDataProvider._ego_vehicle_route.clear()
        CarlaDataProvider._egos = {}
        CarlaDataProvider._carla_actor_pool = {}
        CarlaDataProvider._scenario_actors = {}
        CarlaDataProvider._ego_nearby_agents = {}
        CarlaDataProvider._CBV_nearby_agents = {}
        CarlaDataProvider._vehicle_info = {}
        CarlaDataProvider._client = None
        CarlaDataProvider._spawn_points = None
        CarlaDataProvider._spawn_index = 0
        CarlaDataProvider._traffic_random_seed = 0
        CarlaDataProvider._rng = random.RandomState(CarlaDataProvider._traffic_random_seed)
