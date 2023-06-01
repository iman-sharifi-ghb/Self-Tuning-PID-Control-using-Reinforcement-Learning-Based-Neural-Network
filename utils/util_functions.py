import numpy as np


class utils(object):
    def __init__(self):
        super(utils, self).__init__()
        self.parameters()

    def parameters(self):
        self.m = 0.65;
        self.g = 9.81;
        self.l = 0.23;

        self.Jx = 7.5e-3;
        self.Jy = 7.5e-3;
        self.Jz = 1.3e-2;

        self.b = 3.13e-5;
        self.d = 7.5e-5;

        self.a1 = (self.Jy - self.Jz) / self.Jx;
        self.a2 = (self.Jz - self.Jx) / self.Jy;
        self.a3 = (self.Jx - self.Jy) / self.Jz;

        self.b1 = self.l / self.Jx;
        self.b2 = self.l / self.Jy;
        self.b3 = 1 / self.Jz;


def parameters():
    m = 0.65;
    g = 9.81;
    l = 0.23;

    Jx = 7.5e-3;
    Jy = 7.5e-3;
    Jz = 1.3e-2;

    b = 3.13e-5;
    d = 7.5e-5;

    a1 = (Jy - Jz) / Jx;
    a2 = (Jz - Jx) / Jy;
    a3 = (Jx - Jy) / Jz;

    b1 = l / Jx;
    b2 = l / Jy;
    b3 = 1 / Jz;

    return m, g, l, Jx, Jy, Jz, b, d, a1, a2, a3, b1, b2, b3


def RLS(phi, y, teta, P):
    K = P * phi * np.linalg.inv(1 + phi.T * P * phi)
    teta += K * (y - phi.T * teta)
    P = (np.eye(P.shape[0]) - K * phi.T) * P
    return teta, P


def Saturation(x, min_, max_):
    return max(min(x, max_), min_)


def Reward(S, S_desired, S_dot=0, S_dot_desired=0):
    return -(S - S_desired) ** 2 - 0.1 * (S_dot - S_dot_desired) ** 2


def reset_parameters(actor, critic, init_type="uniform"):
    if init_type == "uniform":
        actor.hidden1.weight.data.uniform_(-1e-3, 1e-3)
        actor.hidden1.bias.data.uniform_(-1e-3, 1e-3)
        actor.hidden2.weight.data.uniform_(-1e-3, 1e-3)
        actor.hidden2.bias.data.uniform_(-1e-3, 1e-3)
        actor.out1.weight.data.uniform_(-1e-3, 1e-3)
        actor.out1.bias.data.uniform_(-1e-3, 1e-3)

        actor.fc1.weight.data.uniform_(-1e-3, 1e-3)
        actor.fc1.bias.data.uniform_(-1e-3, 1e-3)
        actor.out2.weight.data.uniform_(-1e-3, 1e-3)
        actor.out2.bias.data.uniform_(-1e-3, 1e-3)

        critic.hidden1.weight.data.uniform_(-1e-3, 1e-3)
        critic.hidden1.bias.data.uniform_(-1e-3, 1e-3)
        critic.hidden2.weight.data.uniform_(-1e-3, 1e-3)
        critic.hidden2.bias.data.uniform_(-1e-3, 1e-3)
        critic.out.weight.data.uniform_(-1e-3, 1e-3)
        critic.out.bias.data.uniform_(-1e-3, 1e-3)

    if init_type == "fill":
        actor.hidden1.weight.data.fill_(-1e-3, 1e-3)
        actor.hidden1.bias.data.fill_(-1e-3, 1e-3)
        actor.hidden2.weight.data.fill_(-1e-3, 1e-3)
        actor.hidden2.bias.data.fill_(-1e-3, 1e-3)
        actor.out1.weight.data.fill_(-1e-3, 1e-3)
        actor.out1.bias.data.fill_(-1e-3, 1e-3)

        actor.fc1.weight.data.fill_(-1e-3, 1e-3)
        actor.fc1.bias.data.fill_(-1e-3, 1e-3)
        actor.out2.weight.data.fill_(-1e-3, 1e-3)
        actor.out2.bias.data.fill_(-1e-3, 1e-3)

        critic.hidden1.weight.data.fill_(-1e-3, 1e-3)
        critic.hidden1.bias.data.fill_(-1e-3, 1e-3)
        critic.hidden2.weight.data.fill_(-1e-3, 1e-3)
        critic.hidden2.bias.data.fill_(-1e-3, 1e-3)
        critic.out.weight.data.fill_(-1e-3, 1e-3)
        critic.out.bias.data.fill_(-1e-3, 1e-3)

    return actor, critic


def reset_params(agent, init_type="fill"):
    if init_type == "uniform":
        agent.actor_critic.hidden1.weight.data.uniform_(-1e-3, 1e-3)
        agent.actor_critic.hidden2.weight.data.uniform_(-1e-3, 1e-3)
        agent.actor_critic.out1.weight.data.uniform_(-1e-3, 1e-3)
        agent.actor_critic.fc1.weight.data.uniform_(-1e-3, 1e-3)
        agent.actor_critic.fc2.weight.data.uniform_(-1e-3, 1e-3)
        agent.actor_critic.mu.weight.data.uniform_(-1e-3, 1e-3)
        agent.actor_critic.sig.weight.data.uniform_(-1e-3, 1e-3)
        agent.actor_critic.v.weight.data.uniform_(-1e-3, 1e-3)

        agent.actor_critic.hidden1.bias.data.uniform_(-1e-3, 1e-3)
        agent.actor_critic.hidden2.bias.data.uniform_(-1e-3, 1e-3)
        # agent.actor_critic.out1.bias.data.uniform_(-1e-3,1e-3)
        agent.actor_critic.fc1.bias.data.uniform_(-1e-3, 1e-3)
        agent.actor_critic.fc2.bias.data.uniform_(-1e-3, 1e-3)
        agent.actor_critic.mu.bias.data.uniform_(-1e-3, 1e-3)
        agent.actor_critic.sig.bias.data.uniform_(-1e-3, 1e-3)
        agent.actor_critic.v.bias.data.uniform_(-1e-3, 1e-3)

    if init_type == "fill":
        agent.actor_critic.hidden1.weight.data.fill_(0.001)
        agent.actor_critic.hidden2.weight.data.fill_(0.001)
        agent.actor_critic.out1.weight.data.fill_(0.001)
        agent.actor_critic.fc1.weight.data.fill_(0.001)
        agent.actor_critic.fc2.weight.data.fill_(0.001)
        agent.actor_critic.mu.weight.data.fill_(0.001)
        agent.actor_critic.sig.weight.data.fill_(0.001)
        agent.actor_critic.v.weight.data.fill_(0.001)

        agent.actor_critic.hidden1.bias.data.fill_(0.001)
        agent.actor_critic.hidden2.bias.data.fill_(0.001)
        # agent.actor_critic.out1.bias.data.fill_(0.001)
        agent.actor_critic.fc1.bias.data.fill_(0.001)
        agent.actor_critic.fc2.bias.data.fill_(0.001)
        agent.actor_critic.mu.bias.data.fill_(0.001)
        agent.actor_critic.sig.bias.data.fill_(0.001)
        agent.actor_critic.v.bias.data.fill_(0.001)

    return agent


def SignApprox(S, Type, a):
    if Type == 1:
        out = np.sign(S)

    elif Type == 2:
        phi = a
        out = Sat(S / phi, np.array([-1, 1]))

    else:
        coef = a
        out = np.tanh(coef * S)

    return out


def Sat(S, bound):
    if S > np.max(bound):
        out = np.max(bound)

    elif S < np.min(bound):
        out = np.min(bound)

    else:
        out = S;

    return out
