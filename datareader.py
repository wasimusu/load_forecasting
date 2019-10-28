import pandas as pd
import os
from datarepr import repr_date
import numpy as np
import datetime
from utils import timeit


class DataReader:
    def __init__(self, fname):
        self.fname = fname

        if not os.path.exists(self.fname):
            raise ValueError("File {} does not exist".format(fname))

        # self.df = pd.DataFrame(pd.read_csv(self.fname, dtype={'AEP_MW': int}))
        self.df = pd.DataFrame(pd.read_csv(self.fname))

    def get_data(self):
        # returns datetime and power usage.
        # have to represent datetime using lots of cosines.
        # cosines of day, month, hour,
        N = 5000
        X = self.df[self.df.columns[0]]
        Y = np.asarray(self.df[self.df.columns[1]][:N], dtype=np.float)

        X = [repr_date(date) for date in X[:N]]
        X = np.asarray(X)
        return X, Y


@timeit
def f1():
    fname = "data/AEP_hourly.csv"
    datareader = DataReader(fname)
    X, Y = datareader.get_data()


if __name__ == '__main__':
    f1()
