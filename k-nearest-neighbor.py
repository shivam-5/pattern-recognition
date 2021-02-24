import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
data = iris.data
target = iris.target
n = 1

knn1 = KNeighborsClassifier(1)
knn1.fit(data, target)

knn5 = KNeighborsClassifier(5)
knn5.fit(data, target)

X = np.array([[6.3, 2.5, 2.1, 1.0],
              [7.5, 3.1, 5.9, 1.8],
              [7.6, 2.0, 5.3, 1.0],
              [5.6, 4.4, 2.0, 1.0],
              [6.8, 2.7, 2.6, 0.3]])

k1 = knn1.predict(X)
k5 = knn5.predict(X)
print(k1)
print(k5)