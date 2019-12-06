from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from datareader import DataReader, window
import os


def boostedRegressor(X, Y, location, window_size):
    trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, shuffle=False)

    D_train = xgb.DMatrix(trainX, label=trainY)
    D_test = xgb.DMatrix(testX, label=testY)

    param = {
        'eta': 0.25,
        'max_depth': 10,
        'objective': 'reg:squarederror',
    }
    steps = 100  # The number of training iterations

    model = xgb.train(param, D_train, steps)

    preds = model.predict(D_test)
    print("Y:\t\t", testY[:10])
    print("Pred Y : ", np.round(preds[:10], 2))

    plt.title(
        "Scatterplot between actual & predicted load : {} - boosting - {} days ahead".format(location, window_size))
    plt.xlabel("Actual Y (Actual load)")
    plt.ylabel("Predicted Y (Predicted load)")
    plt.scatter(testY, preds)
    plt.show()

    plt.title("Actual & predicted load : {} - boosting - {} days ahead".format(location, window_size))
    plt.plot(list(range(len(preds))), testY, label='Actual Load')
    plt.plot(list(range(len(preds))), preds, label='Predicted Load')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    fname = "data/AEP_daily.csv"
    datareader = DataReader(fname, encoding='Plain', sample_size=20000)  # works
    features, Y = datareader.get_data()

    window_size = 7
    features = features[:-window_size]
    X, Y = window(Y, window_size)
    X = np.concatenate((X, features), axis=1)
    print(features[0])
    print(features.shape, X.shape, Y.shape, X[0], Y[0])

    location = os.path.split(fname)[1].split(".")[0]
    boostedRegressor(X, Y, location, window_size)
