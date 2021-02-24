import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
data = iris.data
target = iris.target

X = data
M = X.shape[0]
y = target
b = np.ones(X.shape[0])

epoch = 2
a = np.array([0.5, -0.5, 3.5, -2.5, 3.5])
alpha = 0.01

# Normalize
X = np.c_[np.ones(M), X]
for i in range(0, M):
    if y[i] == 1 or y[i] == 2:
        X[i, :] = X[i, :] * -1

# Calculate percentage correct before
correct = 0
for i in range(0, M):
    gX = np.around(np.dot(np.r_[1, data[i, :]], a), 4)
    if y[i] == 0 and gX > 0:
        correct = correct + 1
    elif (y[i] == 1 or y[i] == 2) and gX <= 0:
        correct = correct + 1
pBefore = np.around((correct / M) * 100, 1)

# Linear Discriminant Function calculation
for i in range(0, epoch):
    for ii in range(0, M):
        aTy = np.around(np.dot(X[ii, :], a), 4)
        a = np.around(a + alpha * (b[ii] - aTy) * X[ii, :], 4)
print(np.around(a, 2))

# Calculate percentage correct after
correct = 0
for i in range(0, M):
    gX = np.around(np.dot(np.r_[1, data[i, :]], a), 4)
    if y[i] == 0 and gX > 0:
        correct = correct + 1
    elif (y[i] == 1 or y[i] == 2) and gX <= 0:
        correct = correct + 1
pAfter = np.around((correct / M) * 100, 1)

print(pBefore, pAfter)