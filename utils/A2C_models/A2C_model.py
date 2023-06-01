# -*- coding: utf-8 -*-
"""
PID + Actor Critic
1- Continous Action with mu and sigma.
2- Only one Loss function & one Optimizer.

Created on Sun Aug  1 20:47:40 2021
@author: Iman Sharifi
"""
import torch
import torch.nn as nn
import torch as T
import torch.optim as optim
from math import pi
from math import e as E


def calc_logprob(mu_v, var_v, actions_v):
    p1 = - ((mu_v - actions_v) ** 2) / (2 * var_v.clamp(min=1e-3))
    p2 = - torch.log(torch.sqrt(2 * pi * var_v))
    return p1 + p2


class ActorCriticNetwork(nn.Module):
    def __init__(self, state_size, hidden_size, alpha=0.01, K_static=0, K_domain=0, Sigma_domain=0.01):
        super(ActorCriticNetwork, self).__init__()
        action_size = 3
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.K_static = K_static
        self.K_domain = K_domain
        self.Sigma_domain = Sigma_domain

        self.hidden1 = nn.Linear(state_size, hidden_size, bias=True)
        self.hidden2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.out1 = nn.Linear(hidden_size, action_size, bias=False)
        self.u = nn.Linear(action_size, 1, bias=False)
        for params in self.u.parameters():
            params.requires_grad = False
        self.fc1 = nn.Linear(1 + 2, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=True)

        self.mu = nn.Linear(hidden_size, 1)
        self.sig = nn.Linear(hidden_size, 1)
        self.v = nn.Linear(hidden_size, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, x, x_prime):
        x = torch.sigmoid(self.hidden1(x))
        x = torch.sigmoid(self.hidden2(x))

        self.PID_gains = self.K_static + self.K_domain * torch.tanh(self.out1(x))
        # print(self.PID_gains)
        u = self.u(self.PID_gains)
        self.Control_Signal = u

        xs = torch.cat((u, x_prime), dim=1)
        xs = torch.sigmoid(self.fc1(xs))
        xs = torch.sigmoid(self.fc2(xs))

        mu = self.mu(xs)
        sigma = self.Sigma_domain * (1 + T.tanh(self.sig(xs)))  # Sigma must be bounded and positive
        v = self.v(xs)

        self.Mu = mu
        self.Sigma = sigma

        return mu, sigma, v

    def PID(self):
        return self.PID_gains

    def CtrlSignal(self):
        return self.Control_Signal


class NewAgent(object):
    def __init__(self, state_size, hidden_size=5, alpha=0.01, gamma=0.99, K_static=0, K_domain=0, Sigma_domain=0.01):
        self.gamma = gamma
        self.log_probs = None
        self.actor_critic = ActorCriticNetwork(state_size, hidden_size, alpha, K_static, K_domain, Sigma_domain)

    def choose_action(self, input1, input2, yd):
        mu, sigma, v = self.actor_critic.forward(input1, input2)

        if sigma <= torch.tensor([0.01 * mu[0][0]]):
            sigma = torch.tensor([0.01 * mu[0][0]])
        self.sigma = sigma

        action_probs = T.distributions.Normal(mu, sigma)
        probs = action_probs.sample(sample_shape=T.Size([1]))
        self.log_probs = calc_logprob(mu, sigma, probs)  # action_probs.log_prob(probs)
        action = probs  # T.tanh(probs)

        self.e_mu = mu - yd
        self.e_sigma = 0.1 * (self.sigma - abs(self.e_mu))

        return action.item()

    def learn(self, state1, state2, reward, new_state1, new_state2, done=0):
        self.actor_critic.optimizer.zero_grad()

        _, _, critic_value_ = self.actor_critic.forward(new_state1, new_state2)
        _, _, critic_value = self.actor_critic.forward(state1, state2)

        reward = T.tensor(reward, dtype=T.float)
        delta = reward + self.gamma * critic_value_ * (1 - int(done)) - critic_value
        self.delta = delta

        # Individual Losses
        w1, w2, w3, w4 = 1.0, 1.0, 1.0, 1.0
        #         actor_loss  = -self.log_probs * abs(delta)
        actor_loss = w1 * (0.1 + abs(delta)) * self.e_mu ** 2 + w2 * self.e_sigma ** 2
        critic_loss = w3 * delta ** 2
        entropy = w4 * T.sqrt(2 * pi * E * self.sigma.pow(2))

        # Losses
        self.actor_loss = actor_loss
        self.critic_loss = critic_loss
        self.entropy = entropy
        self.Total_Loss = actor_loss + critic_loss + entropy

        # Back Propagation
        (actor_loss + critic_loss + entropy).backward()

        # Update Optimizer
        self.actor_critic.optimizer.step()
