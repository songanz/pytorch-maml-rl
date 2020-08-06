from time import sleep
import numpy as np
import matplotlib.pyplot as plt
import glob

# _before: before adaptation
# _after: after adaptation
# train: before adaptation
# valide: after adaptation

def data_process(input_folder, total_batches=None):
    loss_before = []
    kl_before = []
    loss_after = []
    kl_after = []
    tasks = []
    train_returns = []
    valid_returns = []

    file_list = glob.glob(input_folder + 'policy*')
    if not total_batches:
        total_batches = len(file_list)

    for i in range(total_batches):
        data = np.load(input_folder + 'logs' + str(i))

        loss_before = np.concatenate((loss_before, data[data.files[0]]), axis=0)
        kl_before = np.concatenate((kl_before, data[data.files[1]]), axis=0)
        loss_after = np.concatenate((loss_after, data[data.files[2]]), axis=0)
        kl_after = np.concatenate((kl_after, data[data.files[3]]), axis=0)
        # batch = data[data.files[4]]
        tasks = np.concatenate((tasks, data[data.files[5]]), axis=0)
        # num_interations = data[data.files[6]]

        # train_episodes: after gradient steps
        # valid_episodes: without gradient steps
        train_return = np.average(data[data.files[7]], axis=1)
        train_returns = np.concatenate((train_returns, train_return), axis=0)
        valid_return = np.average(data[data.files[8]], axis=1)
        valid_returns = np.concatenate((valid_returns, valid_return), axis=0)

    # moving average
    n = np.size(train_returns, axis=0)
    m = np.size(loss_before, axis=0)
    train_returns_p = np.zeros(train_returns.size)
    valid_returns_p = np.zeros(valid_returns.size)
    loss_after_p = np.zeros(loss_after.size)
    loss_before_p = np.zeros(loss_before.size)
    for i in range(n-50):
        train_returns_p[i] = np.mean(train_returns[i:i+50])
        valid_returns_p[i] = np.mean(valid_returns[i:i+50])

    for i in range(m-50):
        loss_after_p[i] = np.mean(loss_after[i:i+50])
        loss_before_p[i] = np.mean(loss_before[i:i+50])

    xx = np.linspace(1,n-50, num=n-50)
    yy = np.linspace(1, m - 50, num=m - 50)

    return xx, train_returns_p, valid_returns_p, yy, loss_before_p, loss_after_p

if __name__ == '__main__':

    input_folders = [
                     # 'maml-highway/No_safetyCheck/07152020/',
                     # 'maml-highway/No_safetyCheck/07172020/',
                     # 'maml-highway/No_safetyCheck/first_order_app/07172020/',
                     # 'maml-highway/06272020/',
                     # 'maml-highway/06292020/',
                     # 'maml-highway/No_safetyCheck/07152020/',
                     # 'maml-highway/No_safetyCheck/entropyBonus/07202020/',
                     'maml-highway/optuna-best/08042020/'
                     # 'maml-highway/No_safetyCheck/lr01/07172020/'
                     # 'maml-highway/No_safetyCheck/entropyBonus/lr01/07202020/',
                     # 'maml-highway/No_safetyCheck/lr001/07172020/',
                     # 'maml-highway/No_safetyCheck/entropyBonus/lr001/07202020/'
                     # 'maml-highway/No_safetyCheck/first_order_app/07172020/',
                     # 'maml-highway/No_safetyCheck/entropyBonus/first_order_app/07202020/'
                     ]

    # plot returns of before adaptation and after adaptation
    f = plt.figure(figsize=(10,8))
    for folder in input_folders:
        x, train_returns_plot, valid_returns_plot, _, _, _ = data_process(folder, total_batches=100)

        # plt.plot(x[-100:], train_returns_plot[-150:-50], label=folder + ' before adaptation')
        plt.plot(x, train_returns_plot[:-50], label=folder + ' before adaptation')
        # plt.plot(x[-100:], valid_returns_plot[-150:-50], label=folder + ' after adaptation')
        plt.plot(x, valid_returns_plot[:-50], label=folder + ' after adaptation')

    # x, train_returns_plot, valid_returns_plot, _, _, _ = data_process('maml-highway/06272020/', total_batches=500)
    # plt.plot(x[-100:], valid_returns_plot[-150:-50], label='maml-highway/06272020/' + ' after adaptation')

    plt.title('Returns before and after adaptation')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))
    f.show()

    # plot loss before adaptation and after adaptation
    g = plt.figure(figsize=(10,8))
    for folder in input_folders:
        _,_,_, y, loss_before_plot, loss_after_plot = data_process(folder)

        plt.plot(y, loss_before_plot[:-50], label=folder + ' before adaptation')
        plt.plot(y, loss_after_plot[:-50], label=folder + ' after adaptation')

    plt.title('Loss before and after adaptation')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))
    g.show()

    # sleep(0.1)