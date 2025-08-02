# utils.py

import torch
import torch.nn as nn

def make_mlp(num_units):
    layers = []
    for n in num_units:
        layers.append(nn.LazyLinear(n))
        layers.append(nn.LeakyReLU())
        layers.append(nn.LayerNorm(n))
    return nn.Sequential(*layers)

class GAE(nn.Module):
    def __init__(self, gamma, lmbda):
        super().__init__()
        self.register_buffer("gamma", torch.tensor(gamma))
        self.register_buffer("lmbda", torch.tensor(lmbda))

    def forward(self, reward, terminated, value, next_value):
        num_steps = terminated.shape[1]
        advantages = torch.zeros_like(reward)
        not_done = 1 - terminated.float()
        gae = 0
        for step in reversed(range(num_steps)):
            delta = reward[:, step] + self.gamma * next_value[:, step] * not_done[:, step] - value[:, step]
            gae = delta + self.gamma * self.lmbda * not_done[:, step] * gae
            advantages[:, step] = gae
        returns = advantages + value
        return advantages, returns
