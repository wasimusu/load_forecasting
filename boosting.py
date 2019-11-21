from sklearn.model_selection import train_test_split
from sklearn import datasets
import xgboost as xgb
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt


def boostedClassifier(X, Y):
    trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2)

    D_train = xgb.DMatrix(trainX, label=trainY)
    D_test = xgb.DMatrix(testX, label=testY)

    param = {
        'eta': 0.3,
        'max_depth': 3,
        'objective': 'multi:softprob',
        'num_class': 3}

    steps = 20  # The number of training iterations

    model = xgb.train(param, D_train, steps)

    preds = model.predict(D_test)
    best_preds = np.asarray([np.argmax(line) for line in preds])

    precision = precision_score(testY, best_preds, average='macro')
    recall = recall_score(testY, best_preds, average='macro')
    accuracy = accuracy_score(testY, best_preds)

    print("Precision = {}".format("%.2f" % precision))
    print("Recall = {}".format("%.2f" % recall))
    print("Accuracy = {}".format("%.2f" % accuracy))


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
    plt.xlabel("Actual Y")
    plt.ylabel("Predicted Y")
    plt.scatter(testY, preds)
    plt.show()

    # precision = precision_score(testY, best_preds, average='macro')
    # recall = recall_score(testY, best_preds, average='macro')
    # accuracy = accuracy_score(testY, best_preds)

    # print("Precision = {}".format("%.2f" % precision))
    # print("Recall = {}".format("%.2f" % recall))
    # print("Accuracy = {}".format("%.2f" % accuracy))


if __name__ == '__main__':
    # data = datasets.load_digits()
    # X = data.data
    # Y = data.target

    from datareader import DataReader

    fname = "data/EKPC_hourly.csv"
    # datareader = DataReader(fname, encoding='Plain', sample_size=8500) # works
    datareader = DataReader(fname, encoding='Pair', sample_size=10000)  # works
    X, Y = datareader.get_data()

    print(X.shape)
    print(Y.shape)
    # boostedClassifier(X, Y)
    boostedRegressor(X, Y)
