import numpy as np
from datareader import DataReader


def linear_regeression(X, Y):
    """
    :param X:
    :param Y:
    :return:
    """
    coeffs = np.linalg.lstsq(X, Y, rcond=None)
    print(coeffs)


def f1():
    fname = "data/AEP_hourly.csv"
    datareader = DataReader(fname)
    X, Y = datareader.get_data()
    linear_regeression(X, Y)


if __name__ == '__main__':
    f1()
