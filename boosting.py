from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from datareader import DataReader


def boostedRegressor(X, Y):
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

    plt.title("Scatter plot between actual and predicted Y")
    plt.xlabel("Actual Y (Actual load)")
    plt.ylabel("Predicted Y (Predicted load)")
    plt.scatter(testY, preds)
    plt.show()

    plt.title("Y and pred Y over time")
    plt.plot(list(range(len(preds))), testY, label='Actual Load')
    plt.plot(list(range(len(preds))), preds, label='Predicted Load')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    fname = "data/EKPC_hourly.csv"
    datareader = DataReader(fname, encoding='Plain', sample_size=20000)  # works
    X, Y = datareader.get_data()

    step = 1
    X = np.asarray(Y[:-step]).reshape(-1, 1)
    Y = Y[step:]
    print(X.shape, Y.shape)

    boostedRegressor(X, Y)
