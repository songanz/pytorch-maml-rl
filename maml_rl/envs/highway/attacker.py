import torch as tr
import torch.nn as nn
from torch.autograd import Variable

from . import params


class QNetwork(nn.Module):
    def __init__(self, name='network', hidden_size_IP=100, hidden_size_rest=100, alpha=0.01,
                 state_size=params.CAR_NUM_STATES_ATT, action_size=params.CAR_NUM_ACTIONS, learning_rate=1e-6):
        super(QNetwork, self).__init__()
        self.name = name
        self.alpha = alpha
        self.lr = learning_rate

        self.fc1 = nn.Linear(state_size, hidden_size_IP)
        self.bn1 = nn.BatchNorm1d(hidden_size_IP)

        self.fc2 = nn.Linear(hidden_size_IP, hidden_size_rest)
        self.bn2 = nn.BatchNorm1d(hidden_size_rest)

        self.fc3 = nn.Linear(hidden_size_rest, hidden_size_rest)
        self.bn3 = nn.BatchNorm1d(hidden_size_rest)

        self.Qval = nn.Linear(hidden_size_rest, action_size)

    def forward(self, x):
        x = nn.LeakyReLU(self.alpha)(self.bn1(self.fc1(x)))
        x = nn.LeakyReLU(self.alpha)(self.bn2(self.fc2(x)))
        x = nn.LeakyReLU(self.alpha)(self.bn3(self.fc3(x)))
        x = self.Qval(x)

        if len(x.shape) == 1:
            act = x.max(dim=0)[1]
            Qmax = x.max(dim=0)[0]
        else:
            act = x.max(dim=1)[1]
            Qmax = x.max(dim=1)[0]

        return x, act, Qmax


class attacker(object):
    def __init__(self):
        self.Q_eval_net, self.Q_tar_net = QNetwork(name='eval_net'), QNetwork(name='tar_net')
        self.lr = self.Q_eval_net.lr

        self.loss_fun = nn.MSELoss()
        self.opt = tr.optim.Adam(self.Q_eval_net.parameters(), lr=self.lr)

    def to(self, device):
        self.Q_tar_net.to(device)
        self.Q_eval_net.to(device)

    def learn_step(self, stateBuf, actionBuf, rewards, nextStateBuf, doneBuf, device, r_bonus=None):
        self.Q_eval_net.eval()
        self.Q_tar_net.eval()

        Q_eval, _, _ = self.Q_eval_net(nextStateBuf)
        act_indx = Q_eval.max(dim=1)[1].data.view(-1, 1)
        act_onehot = tr.zeros(params.BATCH_SIZE, params.CAR_NUM_ACTIONS).to(device)
        act_onehot = Variable(act_onehot.scatter_(1, act_indx, 1.0))  # from the Q_eval_net of the next state

        tarQs, _, tarQ_max = self.Q_tar_net(nextStateBuf)

        if not r_bonus is None:
            rewards = tr.add(Variable(tr.from_numpy(r_bonus).to(device)).float(), rewards)

        if params.DDQN:  # DDQN
            tarQ = rewards + tr.mul((tarQs * act_onehot).sum(dim=1) * doneBuf, params.GAMMA)
        else:  # DQN
            tarQ = rewards + tr.mul(tarQ_max * doneBuf, params.GAMMA)

        self.Q_eval_net.train()

        Qs, _, _ = self.Q_eval_net(stateBuf)
        a = actionBuf.long().data.view(-1,1)
        a_onehot_state = tr.zeros(params.BATCH_SIZE, params.CAR_NUM_ACTIONS).to(device)
        a_onehot_state = Variable(a_onehot_state.scatter_(1, a, 1.0))
        Q_eval = (Qs*a_onehot_state).sum(dim=1)

        loss = self.loss_fun(Q_eval, tarQ.detach())

        # backpropagate
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss

    def update_target_network(self):
        # copy current_network to target network
        self.Q_tar_net.load_state_dict(self.Q_eval_net.state_dict())
