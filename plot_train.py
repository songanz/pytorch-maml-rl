from time import sleep
import numpy as np
import matplotlib.pyplot as plt

# _before: before adaptation
# _after: after adaptation
# train: before adaptation
# valide: after adaptation

loss_before = []
kl_before = []
loss_after = []
kl_after = []
tasks = []
train_returns = []
valid_returns = []

for i in range(499):
    data = np.load('maml-highway/06272020/logs' + str(i))

    loss_before = np.concatenate((loss_before, data[data.files[0]]), axis=0)
    kl_before = np.concatenate((kl_before, data[data.files[1]]), axis=0)
    loss_after = np.concatenate((loss_after, data[data.files[2]]), axis=0)
    kl_after = np.concatenate((kl_after, data[data.files[3]]), axis=0)
    # batch = data[data.files[4]]
    tasks = np.concatenate((tasks, data[data.files[5]]), axis=0)
    # num_interations = data[data.files[6]]

    # train_episodes: after gradient steps
    # valid_episodes: without gradient steps
    train_return = np.average(data[data.files[7]], axis=0)
    train_returns = np.concatenate((train_returns, train_return), axis=0)
    valid_return = np.average(data[data.files[8]], axis=0)
    valid_returns = np.concatenate((valid_returns, valid_return), axis=0)

# moving average
n = np.size(train_returns, axis=0)
train_returns_plot = np.zeros(train_returns.size)
valid_returns_plot = np.zeros(valid_returns.size)
for i in range(n-50):
    train_returns_plot[i] = np.mean(train_returns[i:i+50])
    valid_returns_plot[i] = np.mean(valid_returns[i:i+50])

x = np.linspace(1,n-50, num=n-50)

plt.plot(x,train_returns_plot[:-50], 'r',label='before adaptation')
plt.plot(x, valid_returns_plot[:-50], 'b', label='after adaptation')
plt.legend(loc="lower right")
plt.show()
# sleep(0.1)