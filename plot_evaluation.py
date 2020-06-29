from time import sleep
import numpy as np
import matplotlib.pyplot as plt

data = np.load('maml-highway-eval/06272020/logs')

tasks = data[data.files[0]]

# train_episodes: after gradient steps
# valid_episodes: without gradient steps
trained_episodes = data[data.files[1]]
valid_episodes = data[data.files[2]]

trained_episodes = np.mean(trained_episodes,axis=1)
valid_episodes = np.mean(valid_episodes,axis=1)

n = np.size(trained_episodes, axis=0)
train_returns_plot = np.zeros(trained_episodes.size)
valid_returns_plot = np.zeros(valid_episodes.size)
for i in range(n-50):
    train_returns_plot[i] = np.mean(trained_episodes[i:i+50])
    valid_returns_plot[i] = np.mean(valid_episodes[i:i+50])

x = np.linspace(1,n-50,num=n-50)

plt.plot(x,train_returns_plot[:-50], 'r', label='before adaptation')
plt.plot(x, valid_returns_plot[:-50], 'b', label='after adaptation')
plt.legend(loc="lower right")
plt.show()

# sleep(0.1)