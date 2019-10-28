import numpy as np
from sklearn.svm import SVR

#
svr_poly = SVR(kernel='poly', C=1e3, degree=3)
svr_lin = SVR(kernel='linear', C=100, gamma='auto')
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
