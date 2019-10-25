import pandas as pd
import os


class DataReader:
    def __init__(self, fname):
        self.fname = fname

        if not os.path.exists(self.fname):
            raise ValueError("File {} does not exist".format(fname))

        self.df = pd.DataFrame(pd.read_csv(self.fname))

    def get_data(self):
        print(self.df.head())
        # returns datetime and power usage.
        # have to represent datetime using lots of cosines.
        # cosines of day, month, hour,
        X = self.df[self.df.columns[0]]
        Y = self.df[self.df.columns[0]]
        return X, Y


if __name__ == '__main__':
    fname = "data/AEP_hourly.csv"
    datareader = DataReader(fname)
    X, Y = datareader.get_data()
    print(type(X[0]))
    print(type(Y[0]))
