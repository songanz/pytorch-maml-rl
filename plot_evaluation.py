from time import sleep
import numpy as np
import matplotlib.pyplot as plt

data = np.load('maml-highway-eval/06222020/logs')

tasks = data[data.files[0]]
trained_episodes = data[data.files[1]]
valid_episodes = data[data.files[2]]

x = np.linspace(1,400,400)

plt.plot(x,np.mean(trained_episodes,axis=1), 'r--', x, np.mean(valid_episodes,axis=1), 'b')
plt.show()

# sleep(0.1)