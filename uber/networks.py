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
    def __init__(self, input_dim, hidden_dim, output_dim, batch_size, num_layers=1, bidiectional=False, dropout=0.125):
        super(LSTMRegression, self).__init__()

        # RNN Parameters
        self.num_layers = num_layers
        self.num_directions = 2 if bidiectional else 1
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.dropout = dropout

        # Can try other variants of the RNN as well.
        self.features = nn.LSTM(input_dim, hidden_dim, bidirectional=bidiectional, num_layers=num_layers)
        self.linear = nn.Linear(hidden_dim * self.num_directions, output_dim)

    def forward(self, x):
        """ Take x in degrees """
        x, _ = self.features(x)
        x = F.dropout(x, p=self.dropout, training=True)
        # relu works better than sigmoid
        # sigmoid promotes a monotonous values in prediction and results in more oscillation while training
        x = F.relu(x)
        x = self.linear(x)
        return x


class Autoencoder(nn.Module):
    """
    Adapted from https://towardsdatascience.com/time-series-in-python-part-3-forecasting-taxi-trips-with-lstms-277afd4f811
    Paper reference : https://arxiv.org/pdf/1709.01907.pdf titled "Deep and Confident Prediction for Time Series at Uber"
    """

    def __init__(self, input_size=1, batch_size=128, dropout=0.5):
        super(Autoencoder, self).__init__()
        self.hidden_size = 512
        self.bi = False
        self.directions = 2 if self.bi else 1
        self.batch_size = batch_size

        self.encoder_layers = 1
        self.encoder = nn.LSTM(input_size=input_size,
                               hidden_size=self.hidden_size,
                               num_layers=self.encoder_layers,
                               dropout=0.5,
                               bidirectional=self.bi,
                               batch_first=False)

        self.decoder_layers = 1
        self.decoder = nn.LSTM(input_size=self.hidden_size,
                               hidden_size=input_size,
                               num_layers=self.decoder_layers,
                               dropout=0.25,
                               bidirectional=self.bi,
                               batch_first=False)

        self.dense = nn.Linear(in_features=self.hidden_size // 4, out_features=1)

        self.dropout = dropout

    def forward(self, x):
        # Encoder
        output, (last_hidden, _) = self.encoder(x)
        # output = F.dropout(output, p=self.dropout, training=True)

        # Decoder
        # last_hidden = F.relu(last_hidden)
        output, state = self.decoder(last_hidden)
        # output = F.dropout(output, p=self.dropout, training=True)

        # Pass through the dense layer to get the x-hat.
        # output = F.sigmoid(output)
        # output = self.dense(state[0].squeeze(0))

        return output



if __name__ == '__main__':
    pass
