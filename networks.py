"""
This program shows demonstrates setting up a RNN / LSTM / GRU with the following configurable parameters:
- number of layers
- bidirectional or not
- relation of number of layers and bidirectionality to the hidden state and output of RNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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

        # relu works better than sigmoid
        # sigmoid promotes a monotonous values in prediction and results in more oscillation while training
        x = F.sigmoid(x)
        x = self.linear(x)
        return x

    def initHidden(self):
        return (torch.zeros(self.num_directions * self.num_layers, 1, self.hidden_dim),
                torch.zeros(self.num_directions * self.num_layers, 1, self.hidden_dim))


class FCRegression(nn.Module):
    def __init__(self, input_dim, batch_size):
        super(FCRegression, self).__init__()

        # Can try other variants of the RNN as well.
        layer1_nodes = 128
        layer2_nodes = 256
        layer3_nodes = 64

        self.features = nn.Sequential(
            nn.Linear(input_dim, layer1_nodes),
            nn.Sigmoid(),
            nn.Linear(layer1_nodes, layer2_nodes),
            nn.Sigmoid(),
            nn.Linear(layer2_nodes, layer3_nodes),
            nn.Sigmoid(),
        )
        self.regress = nn.Linear(layer3_nodes, 1)

    def forward(self, x):
        """ Take x """
        x = self.features(x)
        x = self.regress(x)
        return x


if __name__ == '__main__':
    pass
