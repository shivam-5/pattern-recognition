import numpy as np


def batch_delta_learning(X, y, omega, alpha, epoch):
    m = X.shape[0]
    for i in range(0, epoch):
        old_omega = omega
        y_hat = np.dot(X, omega)
        y_hat[y_hat > 0] = 1
        y_hat[y_hat < 0] = 0
        print(y_hat)
        omega = omega + alpha * np.dot((y - y_hat), np.transpose(X))
        print(omega)
        if np.array_equal(old_omega, omega):
            return omega
        old_omega = omega
    return omega


def sequential_delta_learning(X, y, omega, alpha, epoch):
    m = X.shape[0]
    for i in range(0, epoch):
        old_omega = omega
        last_updated = -1
        for ii in range(0, m):
            y_hat = np.dot(X[ii, :], omega)
            if y_hat > 0:
                y_hat = 1
            else:
                y_hat = 0
            omega = omega + alpha * (y[ii] - y_hat) * X[ii, :]
            if np.array_equal(old_omega, omega) is False:
                last_updated = ii
            elif np.array_equal(old_omega, omega) and last_updated == ii:
                break
        if np.array_equal(old_omega, omega):
            return omega
        old_omega = omega
    return omega


X = np.array([[0, 2], [1, 2], [2, 1], [-3, 1], [-2, -1], [-3, -2]])
y = np.array([1, 1, 1, 0, 0, 0])
epoch = 20
alpha = 1
omega = [1, 0, 0]
X_aug = np.c_[np.ones(X.shape[0]), X]
omega = sequential_delta_learning(X_aug, y, omega, alpha, epoch)
# omega = batch_delta_learning(X_aug, y, omega, alpha, epoch)
print(omega)
