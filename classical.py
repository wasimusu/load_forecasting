import numpy as np
from sklearn.svm import SVR, LinearSVR
from datareader import DataReader
import sklearn


def generate_data(N, sigma):
    """ Generate data with given number of points N and sigma """
    noise = np.random.normal(0, sigma, N)
    X = np.random.uniform(0, 3, N)
    Y = 2 * X ** 2 + 3 * X + 1 + noise  # arbitrary function
    return X, Y


class SVRRegression:
    def __init__(self, kernel_type='linear'):
        self.kernel_type = kernel_type

        # Different variants of SVR
        self.svr_poly = LinearSVR(epsilon=0.1)
        self.svr_lin = SVR(kernel='linear', C=100, gamma='auto')
        self.svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)

        self.svr_dict = {
            "poly": self.svr_poly,
            "linear": self.svr_lin,
            "rbf": self.svr_rbf
        }

        self.svr = self.svr_dict[kernel_type]

    def fit_predict(self, X, Y, split=0.25):
        trainX, testX, trainY, testY = sklearn.model_selection.train_test_split(X, Y, test_size=split, shuffle=False)
        self.svr = self.svr.fit(trainX, trainY)
        y_pred = self.svr.predict(testX)
        error = np.sum(np.square(y_pred - testY)) / testY.shape[0]
        return error


if __name__ == '__main__':
    fname = "data/AEP_hourly.csv"
    datareader = DataReader(fname)
    X, Y = datareader.get_data()

    # svm_lin = SVRRegression(kernel_type='linear')
    # error_lin = svm_lin.fit_predict(X, Y)

    svm_poly = SVRRegression(kernel_type='poly')
    error_poly = svm_poly.fit_predict(X, Y)

    # svm_rbf = SVRRegression(kernel_type='rbf')
    # error_rbf = svm_rbf.fit_predict(X, Y)

    # print(error_lin)
    print(error_poly)
    # print(error_rbf)
