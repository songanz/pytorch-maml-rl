from time import sleep
import numpy as np
import matplotlib.pyplot as plt

# data = np.load('maml-highway/No_safetyCheck_batch100x40/07152020/logs_eval')
# data = np.load('maml-highway/No_safetyCheck_first_order_app/07152020/logs_eval')
# data = np.load('maml-highway/No_safetyCheck_lr01/07152020/logs_eval')
data = np.load('maml-highway/No_safetyCheck_lr001/07152020/logs_eval')

variables = data.files
tasks = data[variables[0]]

# train_episodes: after gradient steps
# valid_episodes: without gradient steps
trained_episodes = data[variables[1]]
valid_episodes = data[variables[2]]

trained_episodes = np.mean(trained_episodes,axis=1)
valid_episodes = np.mean(valid_episodes,axis=1)

for i in range(3, len(variables)):
    j = i-3
    exec("after_%s_gradient_step = data[variables[%i]]" % (j, i))
    exec("after_%s_gradient_step = np.mean(after_%s_gradient_step, axis=1)" % (j, j))

''' figure for both the before adaptation and after adaptation curves of all the evaluation episodes '''
f = plt.figure(1)
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
f.show()


''' figure for the return for different update steps '''
g = plt.figure(2)
returns_for_gradient_steps = []

for i in range(3, len(variables)):
    j = i-3
    exec("returns_for_gradient_steps.append(np.sum(after_%s_gradient_step)/after_%s_gradient_step.size)" % (j, j))

y = np.linspace(0,len(variables)-4,num=len(variables)-3)
plt.plot(y,returns_for_gradient_steps, 'r', label='different steps of gradient')
g.show()

# sleep(0.1)