"""
This program shows demonstrates setting up a RNN / LSTM / GRU with the following configurable parameters:
- number of layers
- bidirectional or not
- relation of number of layers and bidirectionality to the hidden state and output of RNN
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from networks import LSTMRegression, FCRegression
from datareader import DataReader
from datasets import RegressionDataset
from sklearn.model_selection import train_test_split

import os

import torch.nn.functional as F


def generate_data(N, sigma):
    """ Generate data with given number of points N and sigma """
    noise = np.random.normal(0, sigma, N)
    X = np.random.uniform(0, 3, N)
    Y = 2 * X ** 2 + 3 * X + 1 + noise  # arbitrary function
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    return X, Y


class Regression(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, batch_size, num_layers=1, bidiectional=False):
        super(Regression, self).__init__()

        # RNN Parameters
        self.num_layers = num_layers
        self.num_directions = 2 if bidiectional else 1
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        self.features = nn.LSTM(input_dim, hidden_dim, bidirectional=bidiectional, num_layers=num_layers)
        self.linear = nn.Linear(hidden_dim * self.num_directions, output_dim)

        self.hidden = self.initHidden()

    def forward(self, x):
        """
        :param x: (sequence_length, batch_size, num_features).
        num_features = num_input_attributes
        """
        x, self.hidden = self.features(x, self.hidden)
        x = F.relu(x)
        x = self.linear(x)
        return x

    def initHidden(self):
        return (torch.zeros(self.num_directions * self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_directions * self.num_layers, self.batch_size, self.hidden_dim))


class Model:
    def __init__(self, input_dim=1, num_layers=1, bidirectional=False, hidden_dim=512, batch_size=8, lr=0.005):
        self.lr = lr
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.filename = "LSTM_hidden_dim-{}-num_layers-{}-dir-".format(hidden_dim, num_layers, bidirectional)

    def train(self, train_iter, test_iter, reuse_model=False):
        # model = LSTMRegression(input_dim=self.input_dim,
        #                        hidden_dim=self.hidden_dim,
        #                        batch_size=self.batch_size,
        #                        num_layers=self.num_layers,
        #                        output_dim=1,
        #                        bidiectional=self.bidirectional)
        #
        # model = FCRegression(input_dim=self.input_dim,
        #                      batch_size=self.batch_size)

        model = Regression(input_dim=self.input_dim,
                           hidden_dim=self.hidden_dim,
                           output_dim=1,
                           batch_size=self.batch_size,
                           num_layers=self.num_layers,
                           bidiectional=self.bidirectional)

        if reuse_model:
            if os.path.exists(self.filename):
                try:
                    model.load_state_dict(torch.load(f=self.filename))
                    print("Retraining saved models")
                except:
                    print("The saved model is not compatible. Starting afresh.")

        # model = FCRegression(self.input_dim, self.batch_size)
        print("Model : ", model)
        print("Batch size : ", self.batch_size)
        criterion = nn.MSELoss()
        optimizer = optim.SGD(params=model.parameters(), lr=self.lr, momentum=0.0, weight_decay=0.00)

        # Training parameters
        num_epoch = 200

        for epoch in range(num_epoch):
            epoch_loss = 0
            for i, [inputs, labels] in enumerate(train_iter):
                inputs = torch.tensor(inputs).float().reshape(1, self.batch_size, -1)
                labels = torch.tensor(labels).float().reshape(-1, 1)
                output = model(inputs)

                model.zero_grad()
                loss = criterion(output, labels)
                loss.backward(retain_graph=True)
                optimizer.step()

                epoch_loss += loss

            print(epoch, "Training loss : ", "%.2f" % epoch_loss.item())
            self.compute_loss(dataiter=test_iter, model=model, criterion=criterion)

            # Save the model every ten epochs
            if (epoch + 1) % 2 == 0:
                torch.save(model.state_dict(), f=self.filename)

    def compute_loss(self, dataiter, model, criterion):
        epoch_loss = 0
        for i, [inputs, labels] in enumerate(dataiter):
            inputs = torch.tensor(inputs).float().reshape(1, self.batch_size, -1)
            labels = torch.tensor(labels).float().reshape(-1, 1)
            output = model(inputs)

            loss = criterion(output, labels)
            epoch_loss += loss.item()

            # Print epoch loss and do manual evalutation
            if i == len(dataiter) - 1:
                print("Epoch Loss : {}".format("%.2f" % epoch_loss))
                with torch.no_grad():
                    output = model(inputs)[:8]
                    output = np.round(output.data.numpy(), 2).reshape(-1)[:8]
                    labels = np.round(labels.data.numpy()[:8], 2).reshape(-1)
                    print("{}\n{}\n\n".format(labels, output))

    def predict(self):
        pass


if __name__ == '__main__':
    # Parameters of the model
    # You can change any of the parameters and expect the network to run without error
    num_layers = 1
    bidirectional = False
    hidden_dim = 512  # 512 worked best so far
    batch_size = 64
    learning_rate = 0.0001  # 0.05 results in nan

    # X, Y = generate_data(batch_size * 8, 1)

    # Worst convergence after using Plain encoding.
    # Sine encoding :
    fname = "data/AEP_hourly.csv"
    datareader = DataReader(fname, encoding='Sine', sample_size=batch_size * 124)
    X, Y = datareader.get_data()

    input_dim = len(X[0])
    print("Input dim : ", input_dim)
    trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.25)
    del X, Y

    train_dataset = RegressionDataset(inputs=trainX, labels=trainY)
    test_dataset = RegressionDataset(inputs=testX, labels=testY)

    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    print("Batches : ", len(train_iter))
    model = Model(input_dim=input_dim,
                  num_layers=num_layers,
                  bidirectional=bidirectional,
                  hidden_dim=hidden_dim,
                  batch_size=batch_size,
                  lr=learning_rate)

    model.train(train_iter, test_iter, reuse_model=True)
