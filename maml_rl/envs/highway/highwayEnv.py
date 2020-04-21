import numpy as np
import gym

from gym import spaces
from gym.utils import seeding

from .utils_attack import *
from .attacker import *



class HighwayEnv(gym.Env):
    def __init__(self, low_cars=10, high_cars=30, low_att=0, high_att=3, device='cpu', task={}):
        super(HighwayEnv, self).__init__()

        self.num_action = 8
        self.num_state = 20
        self.num_attacker_state = 24

        self.action_space = spaces.Discrete(self.num_action)
        self.observation_space = spaces.Box(low=-1.0,
            high=1.0, shape=(self.num_state,), dtype=np.float32)


        # env init
        self.envCars = []

        # seed
        self.np_random = seeding.np_random()
        self.seed()

        # for task
        self._task = task
        self._num_attacker = task.get('num_attacker',0)
        self._num_total_car = task.get('num_total_car',20)

        # task sampler parameter
        self.low_cars = low_cars  # lower bound for the number of total cars
        self.high_cars = high_cars
        self.low_att = low_att  # lower bound for the number of attackers in the environment
        self.high_att = high_att

        # attacker's policy network
        self.device = device
        self.attacker_agent = attacker()
        self.attacker_agent.to(self.device)
        pathname_trained_Att = os.path.abspath('./maml_rl/envs/highway/AttackQNet.pth.tar')
        saved_net_Att = tr.load(pathname_trained_Att)
        self.attacker_agent.Q_eval_net.load_state_dict(saved_net_Att['Q_eval_net'])
        self.attacker_agent.Q_eval_net.eval()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def sample_tasks(self, num_tasks):
        num_attackers = self.np_random.randint(self.low_att, self.high_att, size=(num_tasks,))
        num_total_cars = self.np_random.randint(self.low_cars, self.high_cars, size=(num_tasks,))

        tasks = [{'num_attacker':num_attacker, 'num_total_car': num_total_car}
                 for (num_attacker, num_total_car) in zip(num_attackers, num_total_cars)]

        return tasks

    def reset_task(self, task):
        # reset the task specific parameter, the true reset will be done in `reset`
        # together with the `sample_tasks` function
        self._task = task
        self._num_attacker = task['num_attacker']
        self._num_total_car = task['num_total_car']

    def step(self, action):

        return observation, reward, done, {'task': self._task}

    def reset(self):
        # reset the environement
        self.envCars = initCarsAtt(self._num_total_car, num_attacker=self._num_attacker)
        idC = driveFuncs.getAffordInd(self.envCars)

        return observation

    def render(self, mode='human'):

