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
from networks import LSTMRegression, FCRegression


def generate_data(N, sigma):
    """ Generate data with given number of points N and sigma """
    noise = np.random.normal(0, sigma, N)
    X = np.random.uniform(0, 3, N)
    Y = 2 * X ** 2 + 3 * X + 1 + noise  # arbitrary function
    return X, Y


class Model:
    def __init__(self, input_dim=1, num_layers=1, bidirectional=False, hidden_dim=512, batch_size=8, lr=0.005):
        self.lr = lr
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_layers = num_layers

    def train(self):
        model = LSTMRegression(self.input_dim,
                               self.hidden_dim,
                               output_dim=1,
                               batch_size=self.batch_size,
                               num_layers=self.num_layers,
                               bidiectional=self.bidirectional)

        model = FCRegression(self.input_dim, self.batch_size)
        print("Model : ", model)
        criterion = nn.MSELoss()
        optimizer = optim.SGD(params=model.parameters(), lr=self.lr)

        # Training parameters
        num_epoch = 1000

        for epoch in range(num_epoch):
            inputs, labels = generate_data(N=batch_size, sigma=0)
            inputs = torch.tensor(inputs).reshape(-1, 1).float().unsqueeze(0)
            labels = torch.tensor(labels).reshape(-1, 1).float()

            output = model(inputs)

            model.zero_grad()
            loss = criterion(output, labels)
            loss.backward(retain_graph=True)
            optimizer.step()

            if epoch % 100 == 0:
                print("Epoch  {}\tLoss : {}".format(epoch, "%.2f" % loss.item()))

                # Test the model
                input, labels = generate_data(N=batch_size, sigma=0)
                inputs = torch.tensor(input).reshape(-1, 1).float().unsqueeze(0)

                with torch.no_grad():
                    outputs = model(inputs)
                    inputs = torch.tensor([input]).reshape(-1, 1).float().unsqueeze(0)
                    outputs = np.round(outputs.view(1, -1).detach().numpy(), 2)
                    print(np.round(input, 2), '\n',
                          np.round(labels, 2), '\n', outputs, '\n\n')

    def predict(self):
        pass


if __name__ == '__main__':
    # Parameters of the model
    # You can change any of the parameters and expect the network to run without error
    input_dim = 1
    num_layers = 1
    bidirectional = False
    hidden_dim = 512
    batch_size = 8

    model = Model(input_dim=input_dim,
                  num_layers=num_layers,
                  bidirectional=bidirectional,
                  hidden_dim=hidden_dim,
                  batch_size=batch_size,
                  lr=0.005)
    model.train()
