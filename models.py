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


class Model:
    def __init__(self, input_dim=1, num_layers=1, bidirectional=False, hidden_dim=512, batch_size=8, lr=0.005):
        self.lr = lr
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_layers = num_layers

    def train(self, train_iter, test_iter):
        model = LSTMRegression(self.input_dim,
                               self.hidden_dim,
                               output_dim=1,
                               batch_size=self.batch_size,
                               num_layers=self.num_layers,
                               bidiectional=self.bidirectional)

        # model = FCRegression(self.input_dim, self.batch_size)
        print("Model : ", model)
        print("Batch size : ", self.batch_size)
        print("Hidden size : ", model.hidden[0].size())
        criterion = nn.MSELoss()
        optimizer = optim.SGD(params=model.parameters(), lr=self.lr)

        # Training parameters
        num_epoch = 100

        for epoch in range(num_epoch):
            epoch_loss = 0
            for i, [inputs, labels] in enumerate(train_iter):
                inputs = torch.tensor(inputs).float().unsqueeze(1)
                labels = torch.tensor(labels).float()
                output = model(inputs)

                model.zero_grad()
                loss = criterion(output, labels)
                loss.backward(retain_graph=True)
                optimizer.step()

                epoch_loss += loss
                if i % 100 == 0:
                    self.compute_loss(dataiter=test_iter, model=model, criterion=criterion)
                    break

    def compute_loss(self, dataiter, model, criterion):
        epoch_loss = 0
        for i, [inputs, labels] in enumerate(dataiter):
            inputs = torch.tensor(inputs).float().unsqueeze(1)
            labels = torch.tensor(labels).float()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()

            # We need the last data sample to generate some outputs and do manual evalutation
            if i == len(dataiter) - 2: break

        print("Epoch Loss : {}".format("%.2f" % epoch_loss))

        inputs, labels = dataiter.__next__()
        inputs = torch.tensor(inputs).float().unsqueeze(1)
        labels = torch.tensor(labels).reshape(-1, 1).float()
        with torch.no_grad():
            outputs = model(inputs)
            outputs = np.round(outputs.view(1, -1).detach().numpy(), 2)
            print(np.round(labels, 2), '\n', outputs, '\n\n')

    def predict(self):
        pass


if __name__ == '__main__':
    # Parameters of the model
    # You can change any of the parameters and expect the network to run without error
    input_dim = 4
    num_layers = 1
    bidirectional = False
    hidden_dim = 128  # 512 worked best so far
    batch_size = 8

    model = Model(input_dim=input_dim,
                  num_layers=num_layers,
                  bidirectional=bidirectional,
                  hidden_dim=hidden_dim,
                  batch_size=batch_size,
                  lr=0.01)

    fname = "data/AEP_hourly.csv"
    datareader = DataReader(fname)
    X, Y = datareader.get_data()
    trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.25)
    del X, Y

    train_dataset = RegressionDataset(inputs=trainX, labels=trainY)
    test_dataset = RegressionDataset(inputs=testX, labels=testY)

    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model.train(train_iter, test_iter)
