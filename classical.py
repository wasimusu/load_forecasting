import numpy as np
from sklearn.svm import SVR, LinearSVR
from datareader import DataReader, window
import sklearn
import matplotlib.pyplot as plt
import os


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

    def fit_predict(self, X, Y, location, split=0.25):
        trainX, testX, trainY, testY = sklearn.model_selection.train_test_split(X, Y, test_size=split, shuffle=False)
        self.svr = self.svr.fit(trainX, trainY)
        preds = self.svr.predict(testX)
        error = np.sum(np.square(preds - testY)) / testY.shape[0]

        plt.title("Actual vs Predicted Load ({}) - {} Window size {}".format(location, "SVR_" + self.kernel_type, window_size))
        plt.xlabel("Actual load)")
        plt.ylabel("Predicted load)")
        plt.scatter(testY, preds)
        plt.show()

        plt.title("Actul vs Predicted Load ({}) - {}  Window size {}".format(location, "SVR_" + self.kernel_type, window_size))
        plt.plot(list(range(len(preds))), testY, label='Actual Load')
        plt.plot(list(range(len(preds))), preds, label='Predicted Load')
        plt.legend()
        plt.show()

        return error


if __name__ == '__main__':
    fname = "data/household.csv"  # Works for household. Boosting does not.
    location = os.path.split(fname)[1].split(".")[0]
    datareader = DataReader(fname, sample_size=200000)
    features, Y = datareader.get_data()

    window_size = 28
    # 7 does not predict higher extreme values
    # 28 does not predict lower extreme values

    features = features[:-window_size]
    X, Y = window(Y, window_size)
    X = np.concatenate((X, features), axis=1)
    print(features.shape, X.shape, Y.shape, X[0], Y[0])

    svm_poly = SVRRegression(kernel_type='poly')
    loss = svm_poly.fit_predict(X, Y, location)

    print("Loss : ", "%.2f" % loss)
