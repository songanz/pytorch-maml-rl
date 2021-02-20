import numpy as np
from gym.utils import seeding
from gym import spaces

from maml_rl.envs.IDM_Mobil import utils
from maml_rl.envs.IDM_Mobil.envs.common.abstract import AbstractEnv
from maml_rl.envs.IDM_Mobil.envs.common.action import Action, action_factory
from maml_rl.envs.IDM_Mobil.envs.common.observation import observation_factory
from maml_rl.envs.IDM_Mobil.road.road import Road, RoadNetwork
from maml_rl.envs.IDM_Mobil.vehicle.controller import ControlledVehicle

def action_map(action):
    if 1 / 3 <= action[0] < 1:  # turn right: R
        discrete_action = 2

    elif -1 <= action[0] < -1 / 3:  # turn left: L
        discrete_action = 0

    else:  # no lane change: M
        if action[1] < -0.3:  # brake
            discrete_action = 4
        elif 0 <= action[1] < 0.5:  # maintain
            discrete_action = 1
        else:  # accelerate
            discrete_action = 3

    return discrete_action

class IDM_MOBIL(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    RIGHT_LANE_REWARD = 0.1
    """The reward received when driving on the right-most lanes, linearly mapped to zero for other lanes."""

    HIGH_SPEED_REWARD = 0.4
    """The reward received when driving at full speed, linearly mapped to zero for lower speeds according to config["reward_speed_range"]."""

    LANE_CHANGE_REWARD = 0
    """The reward received at each lane change action."""

    VEHICLES_TYPE_LIST = ["maml_rl.envs.IDM_Mobil.vehicle.behavior.AggressiveVehicle",
                          "maml_rl.envs.IDM_Mobil.vehicle.behavior.DefensiveVehicle",
                          "maml_rl.envs.IDM_Mobil.vehicle.behavior.IDMVehicle"]

    def __init__(self, low_cars=30, high_cars=50, n_tasks=4, randomize_tasks=True):

        self.low_cars = low_cars
        self.high_cars = high_cars
        self.n_tasks = n_tasks
        self.np_random = seeding.np_random()
        self.seed()

        # for debuging
        goals = [{'vehicles_density': 1, 'vehicles_count': 50, 'num_diff_tpyes': [25,25,0]},
                 {'vehicles_density': 1, 'vehicles_count': 50, 'num_diff_tpyes': [50,0,0]},
                 {'vehicles_density': 1, 'vehicles_count': 50, 'num_diff_tpyes': [0,50,0]},
                 {'vehicles_density': 1, 'vehicles_count': 50, 'num_diff_tpyes': [0,0,50]}]

        if randomize_tasks:
            self._goals = self.sample_tasks(self.n_tasks)
            self._goal = self._goals[0]
        else:
            self._goals = goals
            self._goal = self._goals[0]

        super().__init__()  # will call default_config

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action = action_map(action)
        return super(IDM_MOBIL, self).step(action)

    def default_config(self) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 3,
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 20,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "num_diff_tpyes": [0,0,50],  # How many cars in each type in the environment
            "collision_reward": -1,  # The reward received when colliding with a vehicle.
            "reward_speed_range": [28, 30],
            "offroad_terminal": False
        })
        return config

    def sample_tasks(self, num_tasks):

        vehicles_density_s = self.np_random.random_sample(size=(num_tasks,)) + 0.5  # from 0.5 to 1.5
        vehicles_count_s = self.np_random.randint(self.low_cars, self.high_cars, size=(num_tasks,))
        num_diff_tpyes_s = []
        for i in range(num_tasks):
            _sum = vehicles_count_s[i]
            num_types = len(self.VEHICLES_TYPE_LIST)
            num_diff_tpyes = self.np_random.multinomial(_sum, np.ones(num_types)/num_types, size=1)[0]
            num_diff_tpyes_s.append(num_diff_tpyes)

        tasks = [{'vehicles_density': vehicles_density,
                  'vehicles_count': vehicles_count,
                  'num_diff_tpyes': num_diff_tpyes}
                 for (vehicles_density, vehicles_count, num_diff_tpyes)
                 in zip(vehicles_density_s, vehicles_count_s, num_diff_tpyes_s)]

        return tasks

    def get_all_task_idx(self):
        return range(len(self._goals))

    def reset_task(self, task) -> None:
        self._goal = task
        self.config.update(self._goal)
        self.reset()  # will call define_spaces() and _reset()

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def define_spaces(self) -> None:
        self.observation_type = observation_factory(self, self.config["observation"])
        self.action_type = action_factory(self, self.config["action"])
        self.observation_space = self.observation_type.space()
        # action_space must be continuous for the SAC backbone in this repo
        # Therefore write a action map to get the right discrete number
        self.num_action = 2
        self.action_space = spaces.Box(low=-1.0,
            high=1.0, shape=(self.num_action,), dtype=np.float32)

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"]),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        self.controlled_vehicles = []
        for _ in range(self.config["controlled_vehicles"]):
            vehicle = self.action_type.vehicle_class.create_random(self.road,
                                                                   speed=25,
                                                                   lane_id=self.config["initial_lane_id"],
                                                                   spacing=self.config["ego_spacing"])
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

        for i in range(len(self.VEHICLES_TYPE_LIST)):
            vehicles_type = utils.class_from_path(self.VEHICLES_TYPE_LIST[i])
            for _ in range(self.config["num_diff_tpyes"][i]):
                self.road.vehicles.append(
                    vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"]))

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        It will be called in the super(IDM_MOBIL, self).step(action) in def step
        :param action: the last action performed (discrete)
        :return: the corresponding reward
        """
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
        reward = \
            + self.config["collision_reward"] * self.vehicle.crashed \
            + self.RIGHT_LANE_REWARD * lane / max(len(neighbours) - 1, 1) \
            + self.HIGH_SPEED_REWARD * np.clip(scaled_speed, 0, 1)
        reward = utils.lmap(reward,
                          [self.config["collision_reward"], self.HIGH_SPEED_REWARD + self.RIGHT_LANE_REWARD],
                          [0, 1])
        reward = 0 if not self.vehicle.on_road else reward
        return reward

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.vehicle.crashed or \
            self.steps >= self.config["duration"] or \
            (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        return float(self.vehicle.crashed)
