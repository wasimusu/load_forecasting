import pandas as pd
import os
from datarepr import repr_date
import numpy as np
import datetime


def window(X, N):
    """
    Convert input data X into groups on N sliding items
    >>> Y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    >>> window_size = 5
    >>> features, labels = window(X, window_size)

    feature,    label
    [1 2 3 4 5] 6
    [2 3 4 5 6] 7
    [3 4 5 6 7] 8
    [4 5 6 7 8] 9
    [5 6 7 8 9] 10
    """
    output = [X[i:i + N] for i in range(len(X) - N)]
    return np.asarray(output).reshape(-1, N), X[N:]


class DataReader:
    def __init__(self, fname, batch_size=8, encoding='Plain', sample_size=64 * 40):
        """
        :param encoding: Sine, Cosine, Pair(Sine, Cosine), Plain
        """
        self.fname = fname
        self.batch_size = batch_size

        if not os.path.exists(self.fname):
            raise ValueError("File {} does not exist".format(fname))

        # self.df = pd.DataFrame(pd.read_csv(self.fname, dtype={'AEP_MW': int}))
        self.df = pd.DataFrame(pd.read_csv(self.fname))
        self.iter_count = 0

        N = sample_size
        X = self.df[self.df.columns[0]]
        self.Y = np.asarray(self.df[self.df.columns[1]][:N], dtype=np.float)

        nX = []
        nY = []
        for date, y in zip(X[:N], self.Y):
            try:
                nX.append(repr_date(date, type=encoding)[-3:])
                nY.append(y)
            except:
                pass

        X = nX
        self.Y = np.asarray(nY, dtype=np.float)

        # X = [repr_date(date, type=encoding)[-3:] for date in X[:N]]
        self.X = np.asarray(X)

        self.num_batches = len(self.Y) // self.batch_size

        # Trim data that can't be fit into batches
        total = self.num_batches * self.batch_size
        self.X = self.X[:total]
        self.Y = self.Y[:total]

    def get_data(self):
        # returns datetime and power usage.
        # have to represent datetime using lots of cosines/sines/plain values
        # cosines of day, month, hour,
        return self.X, self.Y

    def __next__(self):
        """ Outputs a batch of train and test data """
        start = self.iter_count * self.batch_size
        end = start + self.batch_size
        self.iter_count += 1

        # Start over again
        if self.iter_count + 1 == self.num_batches or end - start < self.batch_size:
            self.iter_count = 0
        return self.X[start:end], self.Y[start:end]

    def __len__(self):
        return self.num_batches


if __name__ == '__main__':
    fname = "data/household.csv"
    datareader = DataReader(fname, encoding='Pair')
    for i in range(len(datareader)):
        X, Y = datareader.__next__()
        print(i, X.shape, Y.shape)
