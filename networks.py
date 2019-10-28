"""
This program shows demonstrates setting up a RNN / LSTM / GRU with the following configurable parameters:
- number of layers
- bidirectional or not
- relation of number of layers and bidirectionality to the hidden state and output of RNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class LSTMRegression(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, batch_size, num_layers=1, bidiectional=False):
        super(LSTMRegression, self).__init__()

        # RNN Parameters
        self.num_layers = num_layers
        self.num_directions = 2 if bidiectional else 1
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        # Can try other variants of the RNN as well.
        self.features = nn.LSTM(input_dim, hidden_dim, bidirectional=bidiectional, num_layers=num_layers)
        self.linear = nn.Linear(hidden_dim * self.num_directions, output_dim)

        self.hidden = self.initHidden()

    def forward(self, x):
        """ Take x in degrees """
        x, self.hidden = self.features(x, self.hidden)
        x = F.relu(x)
        x = self.linear(x)
        return x

    def initHidden(self):
        return (torch.zeros(self.num_directions * self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_directions * self.num_layers, self.batch_size, self.hidden_dim))


class FCRegression(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, batch_size, num_layers=1):
        super(FCRegression, self).__init__()

        # RNN Parameters
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        # Can try other variants of the RNN as well.
        layer1_nodes = 128
        layer2_nodes = 256
        layer3_nodes = 64
        self.input = nn.Linear(input_dim, layer1_nodes)
        self.hidden1 = nn.Linear(layer1_nodes, layer2_nodes)
        self.hidden2 = nn.Linear(layer2_nodes, layer3_nodes)
        self.regress = nn.Linear(layer3_nodes, output_dim)

    def forward(self, x):
        """ Take x in degrees """
        x = self.input(x)
        x = torch.sigmoid(x)

        x = self.hidden1(x)
        x = torch.sigmoid(x)

        x = self.hidden2(x)
        x = torch.sigmoid(x)

        x = F.relu(x)
        x = self.linear(x)
        return x


if __name__ == '__main__':
    pass
