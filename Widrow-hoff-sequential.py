import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
data = iris.data
target = iris.target

# Widrow-Hoff Sequential
X = np.array([[0, 2], [1, 2], [2, 1], [-3, 1], [-2, -1], [-3, -2]])
y = np.array([1, 1, 1, -1, -1, -1])
b = np.array([1.0, 2.5, 1.5, 1.5, 1.5, 1.0])

M = X.shape[0]

epoch = 2
a = np.array([1, 0, 0])
alpha = 0.1

# Normalize
X = np.c_[np.ones(M), X]
for i in range(0, M):
    if y[i] == -1:
        X[i, :] = X[i, :] * -1

for i in range(0, epoch):
    for ii in range(0, M):
        aTy = np.around(np.dot(X[ii, :], a), 4)
        a = np.around(a + alpha * (b[ii] - aTy) * X[ii, :], 4)
        print(np.around(aTy, 2), np.around(a, 2))
