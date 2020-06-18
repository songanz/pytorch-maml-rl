import gym
import maml_rl.envs
from time import sleep
import torch as tr
import yaml
from maml_rl.utils.helpers import get_policy_for_env

if __name__ == '__main__':
    env = gym.make('Highway-v0')

    observation = env.reset()
    for i in range(100):
        env.render()

        yaml_path = './configs/maml/highway.yaml'

        with open(yaml_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        agent = get_policy_for_env(env,
                                   hidden_sizes=config['hidden-sizes'],
                                   nonlinearity=config['nonlinearity'])

        path = './maml-highway/policy.th'
        agent.load_state_dict(tr.load(path))
        agent.eval()

        observations_tensor = tr.from_numpy(observation)
        pi = agent(observations_tensor)
        actions_tensor = pi.sample()
        action = actions_tensor.cpu().numpy()


        observation, reward, done, info = env.step(action)
        sleep(0.01)

    env.render(close=True)
    env.close()
