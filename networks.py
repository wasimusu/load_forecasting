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

        self.hidden = self.initHidden()

    def forward(self, x):
        """ Take x in degrees """
        x, self.hidden = self.features(x, self.hidden)
        x = F.dropout(x, p=self.dropout, training=True)
        # relu works better than sigmoid
        # sigmoid promotes a monotonous values in prediction and results in more oscillation while training
        x = F.relu(x)
        x = self.linear(x)
        return torch.squeeze(x)

    def initHidden(self):
        return (torch.zeros(self.num_directions * self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_directions * self.num_layers, self.batch_size, self.hidden_dim))


class GRURegression(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, batch_size, num_layers=1, bidiectional=False):
        super(GRURegression, self).__init__()

        # RNN Parameters
        self.num_layers = num_layers
        self.num_directions = 2 if bidiectional else 1
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        # Can try other variants of the RNN as well.
        self.features = nn.GRU(input_dim, hidden_dim, bidirectional=bidiectional, num_layers=num_layers)
        self.linear = nn.Linear(hidden_dim * self.num_directions, output_dim)

        self.hidden = self.initHidden()

    def forward(self, x):
        """ Take x in degrees """
        x, self.hidden = self.features(x, self.hidden)

        # relu works better than sigmoid
        # sigmoid promotes a monotonous values in prediction and results in more oscillation while training
        x = F.relu(x)
        x = self.linear(x)
        return x

    def initHidden(self):
        return torch.zeros(self.num_directions * self.num_layers, self.batch_size, self.hidden_dim)


class FCRegression(nn.Module):
    """
    This produces same value for all the inputs to reduce MSE.
    """

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


class LSTMAutoencoder(nn.Module):
    """
    Adapted from https://towardsdatascience.com/time-series-in-python-part-3-forecasting-taxi-trips-with-lstms-277afd4f811
    Paper reference : https://arxiv.org/pdf/1709.01907.pdf titled "Deep and Confident Prediction for Time Series at Uber"
    """

    def __init__(self, config):
        super(LSTMAutoencoder, self).__init__()
        self.hidden_size = 128
        self.bi = False
        self.lstm1 = nn.LSTM(config.get('features'), self.hidden_size, 1, dropout=0.1, bidirectional=self.bi,
                             batch_first=True)
        self.lstm2 = nn.LSTM(self.hidden_size, self.hidden_size // 4, 1, dropout=0.1, bidirectional=self.bi,
                             batch_first=True)
        self.dense = nn.Linear(self.hidden_size // 4, config.get('forecast_horizon'))
        self.loss_fn = nn.MSELoss()
        self.batch_size = config.get('batch_size')

    def forward(self, x):
        # Encoder
        output, _ = self.lstm1(x)
        output = F.dropout(output, p=0.5, training=True)

        # Decoder
        output, state = self.lstm2(output)
        output = F.dropout(output, p=0.5, training=True)  # This should act as the output of Autoencoder

        # Numerical Predictor
        output = self.dense(state[0].squeeze(0))  # Why is it using state but not output. Output of first encoder ?

        return output


class Autoencoder(nn.Module):
    """
    Adapted from https://towardsdatascience.com/time-series-in-python-part-3-forecasting-taxi-trips-with-lstms-277afd4f811
    Paper reference : https://arxiv.org/pdf/1709.01907.pdf titled "Deep and Confident Prediction for Time Series at Uber"
    """

    def __init__(self, input_size=1, batch_size=128, dropout=0.5):
        super(Autoencoder, self).__init__()
        self.hidden_size = 128
        self.bi = False

        self.encoder = nn.LSTM(input_size=input_size,
                               hidden_size=self.hidden_size,
                               num_layers=1, dropout=0.1,
                               bidirectional=self.bi,
                               batch_first=True)

        self.decoder = nn.LSTM(input_size=self.hidden_size,
                               hidden_size=self.hidden_size // 4,
                               num_layers=1,
                               dropout=0.1,
                               bidirectional=self.bi,
                               batch_first=True)

        self.dropout = dropout
        self.batch_size = batch_size
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        # Encoder
        output, _ = self.encoder(x)
        output = F.dropout(output, p=self.dropout, training=True)

        # Decoder
        output, state = self.decoder(output)
        output = F.dropout(output, p=self.dropout, training=True)

        return output


if __name__ == '__main__':
    pass
