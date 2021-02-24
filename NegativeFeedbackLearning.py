import numpy as np


def NegativeFeedbackLearning_basic(x, y, w, alpha, iterations):
    for i in range(0, iterations):
        e = x - np.dot(np.transpose(w), y)
        y = y + alpha * np.dot(w, e)
    return y


# Regulatory Feedback or Decisive Input Modulation
def NegativeFeedbackLearning_stable(x, y, w, epsilon1, epsilon2, iterations):
    # Normalize each row of W to one
    w_mean = np.sum(w, axis=1)
    w_norm = w / w_mean[:, np.newaxis]

    for i in range(0, iterations):
        val1 = np.dot(np.transpose(w), y)
        div = np.where(val1 > epsilon2, val1, epsilon2)
        e = x / div
        y = np.where(y > epsilon1, y, epsilon1) * np.dot(w_norm, e)
    return y


x = np.array([1, 1, 0])
y = np.array([0, 0], dtype=float)
w = np.array([[1, 1, 0], [1, 1, 1]])
alpha = 0.5
epsilon1 = epsilon2 = 0.01

y = NegativeFeedbackLearning_stable(x, y, w, epsilon1, epsilon2, iterations=5)
# y = NegativeFeedbackLearning_basic(x, y, w, alpha, iterations=5)
print(y)