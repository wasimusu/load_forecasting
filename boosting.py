from sklearn.model_selection import train_test_split
from sklearn import datasets
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt


def boostedRegressor(X, Y):
    trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, shuffle=False)

    D_train = xgb.DMatrix(trainX, label=trainY)
    D_test = xgb.DMatrix(testX, label=testY)

    param = {
        'eta': 0.25,
        'max_depth': 40,
        'objective': 'reg:squarederror',
    }

    steps = 100  # The number of training iterations

    model = xgb.train(param, D_train, steps)

    preds = model.predict(D_test)
    print(testY[:10])
    print(np.round(preds[:10], 2))

    plt.title("Scatter plot between actual and predicted Y")
    plt.xlabel("Actual Y (Actual load)")
    plt.ylabel("Predicted Y (Predicted load)")
    plt.scatter(testY, preds)
    plt.show()


if __name__ == '__main__':
    # data = datasets.load_digits()
    # X = data.data
    # Y = data.target

    from datareader import DataReader

    fname = "data/EKPC_hourly.csv"
    datareader = DataReader(fname, encoding='Plain', sample_size=8500) # works
    # datareader = DataReader(fname, encoding='Pair', sample_size=10000)  # works
    X, Y = datareader.get_data()

    # boostedClassifier(X, Y)
    boostedRegressor(X, Y)
