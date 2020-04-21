import gym
import maml_rl.envs
from time import sleep

if __name__ == '__main__':
    env = gym.make('Highway-v0')

    observation = env.reset()
    for i in range(100):
        env.render()
        observation, reward, done, info = env.step(1)
        sleep(0.01)


    env.render(close=True)
    env.close()
