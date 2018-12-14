# -*- coding:utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import models

from lib import torch_util
from model.error import ModelError


class ProbNet(nn.Module):
    def __init__(self, in_size, out_size):
        super(ProbNet, self).__init__()
        self.linear = nn.Linear(in_size, out_size)

    def forward(self, vect):
        return F.log_softmax(self.linear(vect), dim=1)
