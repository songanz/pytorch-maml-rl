from time import sleep
import numpy as np
import matplotlib.pyplot as plt

data = np.load('maml-highway/06262020/logs')
loss_before = data[data.files[0]]
kl_before = data[data.files[1]]
loss_after = data[data.files[2]]
kl_after = data[data.files[3]]
batch = data[data.files[4]]
tasks = data[data.files[5]]
num_interations = data[data.files[6]]
train_returns = data[data.files[7]]
valid_returns = data[data.files[8]]

sleep(0.1)