import gym

from gym import spaces
from gym.utils import seeding

from .utils_attack import *
from .attacker import *
import os


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
        self.safeCheck =  False  # for short horizon safety check of the host vehicle

        # seed
        self.np_random = seeding.np_random()
        self.seed()

        # for task
        self._task = task
        self._num_attacker = task.get('num_attacker',3)
        self._num_total_car = task.get('num_total_car',20)
        self._traffic_density = task.get('traffic_density', 1)

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
        saved_net_Att = tr.load(pathname_trained_Att,  map_location=tr.device('cpu'))
        self.attacker_agent.Q_eval_net.load_state_dict(saved_net_Att['Q_eval_net'])
        self.attacker_agent.Q_eval_net.eval()

        # for render
        self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def sample_tasks(self, num_tasks):
        num_attackers = self.np_random.randint(self.low_att, self.high_att, size=(num_tasks,))
        num_total_cars = self.np_random.randint(self.low_cars, self.high_cars, size=(num_tasks,))
        traffic_densities = self.np_random.random_sample(size=(num_tasks,)) + 0.5  # from 0.5 to 1.5

        tasks = [{'num_attacker':num_attacker,
                  'num_total_car': num_total_car,
                  'traffic_density': traffic_density}
                 for (num_attacker, num_total_car, traffic_density)
                 in zip(num_attackers, num_total_cars, traffic_densities)]

        return tasks

    def reset_task(self, task):
        # reset the task specific parameter, the true reset will be done in `reset`
        # together with the `sample_tasks` function
        self._task = task
        self._num_attacker = task['num_attacker']
        self._num_total_car = task['num_total_car']
        self._traffic_density = task['traffic_density']

    def step(self, action):
        # number of attackers is in the self._num_attacker
        # envCar: 0 (C.T_CAR) -> the target agent of the attackers and the training agent of the maml
        # envCar: 1 ~ 1+self._num_attacker -> the attakcers

        done = False
        # store the state for the current time step
        idC, _ = driveFuncs.getAffordInd(self.envCars)

        if self.safeCheck:
            # safety check
            newAct, rstPrfAct = driveFuncs.safeAct2(action, idC[C.T_CAR, :])

            # change the action according to safety check
            if newAct != action:
                # saftey controller overrides the first choice, set the original action for collision
                # print('safe action')
                action, actOld = newAct, action
                safeAct = True  # safety action was chosen
            else:
                safeAct = False

        if self._num_attacker == 0:

            driveFuncs.setEgoCarAction(self.envCars[C.T_CAR], action)  # set action for the ego car (target)

            driveFuncs.setAction(self.envCars[1:], idC[1:, :])  # set actions for env cars
            driveFuncs.appAction(self.envCars)  # Apply actions

        else:  # env with attackers
            # 0: target / training agent
            # 1 ~ 1 + self._num_attacker: attacker
            attackersID = [ i for i in range(1, 1 + self._num_attacker)]

            # for all the attackers in the env, set its action by the self.attacker_agent policy network
            for i in attackersID:
                # this function is for putting up state for the attacker
                attacker_i_state = getAffIndiOfTarget(self.envCars[i], self.envCars)
                s0_i_attacker = scaleStateAtt(attacker_i_state)

                Q, _, _ = self.attacker_agent.Q_eval_net(
                    Variable(tr.from_numpy(s0_i_attacker).to(self.device), requires_grad=False).float()[None, ...])
                attacker_i_action = int(tr.argmax(Q))

                # constraint on the action of the attacker for more efficient attacking
                lnAtt = idC[i, C.E_LN]
                if lnAtt == params.ROAD_RIGHT_LANE:
                    # on right lane
                    if attacker_i_action == C.A_BR or attacker_i_action == C.A_MR:
                        attacker_i_action = C.A_MM

                elif lnAtt == params.ROAD_LEFT_LANE:
                    # on left lane
                    if attacker_i_action == C.A_BL or attacker_i_action == C.A_ML:
                        attacker_i_action = C.A_MM

                # set the action for each attacker in this env
                # don't be confused by the function name
                driveFuncs.setEgoCarAction(self.envCars[i], attacker_i_action)

            # set action for the target car
            driveFuncs.setEgoCarAction(self.envCars[C.T_CAR], action)

            # set actions for other env cars
            driveFuncs.setAction(self.envCars[1+self._num_attacker:], idC[1+self._num_attacker:, :])

            # Apply actions
            driveFuncs.appAction(self.envCars)


        """ After applied the actions for all envCars, get the observation and reward """
        idC_N, _ = driveFuncs.getAffordInd(self.envCars)  # get next state for all cars

        # check for collision if collision then set rew to some large -ve value
        collision = driveFuncs.checkColi4Ego(idC[C.T_CAR, :], idC_N[C.T_CAR, :], action)
        # Get new scaled ego car state
        eIDC_N = idC_N[0, :]
        s1 = driveFuncs.scaleState(eIDC_N)
        rewV, rewY, rewX, cf_d, maxV, disY, eV, cf_v, midDis = driveFuncs.getRewCmp(eIDC_N)

        if self.safeCheck and safeAct:
            if action != C.A_MM:
                reward = -0.1
            else:
                # don't penalize maintain
                reward = 0
        else:
            reward = (rewV + rewY + rewX) / 3  # scale between -1 to 0
            if reward > -1e-2 and not collision:
                # give a small +ve reward for doing things in a correct manner
                reward = 0.01

        if collision:
            reward = -200  # for collision give more -ve reward
            done = True

        observation = s1

        return observation, reward, done, {'task': self._task}

    def reset(self):

        # reset the environement
        self.envCars = initCarsAtt(self._num_total_car, self._num_attacker, self._traffic_density)
        idC, _= driveFuncs.getAffordInd(self.envCars)
        observation = driveFuncs.scaleState(idC[C.T_CAR, :])

        return observation

    # for HPC: remove this function
    def render(self, mode='human', close=False):
        from gym.envs.classic_control import rendering  # for HPC: comment this line

        # horizotal: x axis; vertical: y axis
        # origin: left bottom corner

        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        # for render
        if self.viewer is None:
            self.viewer = rendering.Viewer(1000, 100)

        # for transform and scale
        scale = 5
        trans = rendering.Transform(translation=(500, 25))

        # clean the canvas each frame
        self.viewer.geoms = []
        # lane boundary
        lane_boundary1 = rendering.Line((-100*scale, 0), (100*scale, 0))
        lane_boundary2 = rendering.Line((-100*scale, 3.6*scale), (100*scale, 3.6*scale))
        lane_boundary3 = rendering.Line((-100*scale, 7.2*scale), (100*scale, 7.2*scale))
        lane_boundary4 = rendering.Line((-100*scale, 10.8*scale), (100*scale, 10.8*scale))

        lane_boundary1.set_color(0,0,0)
        lane_boundary2.set_color(0,0,0)
        lane_boundary3.set_color(0,0,0)
        lane_boundary4.set_color(0,0,0)

        lane_boundary1.add_attr(trans)
        lane_boundary2.add_attr(trans)
        lane_boundary3.add_attr(trans)
        lane_boundary4.add_attr(trans)

        self.viewer.add_geom(lane_boundary1)
        self.viewer.add_geom(lane_boundary2)
        self.viewer.add_geom(lane_boundary3)
        self.viewer.add_geom(lane_boundary4)

        # cars
        for ind in range(len(self.envCars)):
            car = self.envCars[ind]
            l = car.xPos - 2.2
            r = car.xPos + 2.2
            b = car.yPos - 1.1
            t = car.yPos + 1.1

            l *= scale
            r *= scale
            b *= scale
            t *= scale

            if ind in [ i for i in range(1, 1 + self._num_attacker)]:
                car_render = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                car_render.add_attr(trans)
                car_render.set_color(1,0,0)
                self.viewer.add_geom(car_render)
            elif ind == 0:
                car_render = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                car_render.add_attr(trans)
                car_render.set_color(0,0,1)
                self.viewer.add_geom(car_render)
            else:
                car1 = rendering.Line((l, b), (l, t))
                car2 = rendering.Line((l, t), (r, t))
                car3 = rendering.Line((r, t), (r, b))
                car4 = rendering.Line((r, b), (l, b))

                car1.add_attr(trans)
                car2.add_attr(trans)
                car3.add_attr(trans)
                car4.add_attr(trans)

                self.viewer.add_geom(car1)
                self.viewer.add_geom(car2)
                self.viewer.add_geom(car3)
                self.viewer.add_geom(car4)

        return self.viewer.render(return_rgb_array = mode == 'rgb_array')


