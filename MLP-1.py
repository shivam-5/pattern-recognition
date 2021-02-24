import numpy as np
from Activations import Activations

activations = Activations()


def initialize():
    Wji = np.array([[0.5, np.nan],
                    [0.3, -0.7]], dtype='float')
    Wj0 = np.array([0.2, np.nan], dtype='float')
    Wkj = np.array([[0.8, 1.6]], dtype='float')
    Wk0 = np.array([-0.4], dtype='float')
    W = list()
    W.append(np.c_[Wji, Wj0])
    W.append(np.c_[Wkj, Wk0])
    return W


def forward_propagation(X, W):
    input = np.r_[X, 1]
    outputs = list()
    outputs.append([0, input])
    for i in range(0, np.size(W)):
        net = np.dot(np.where(np.isnan(W[i]), 0, W[i]), input)
        if i != np.size(W) - 1:
            Y = activations.tanh(net)     # Hidden layers
        else:
            Y = activations.tanh(net)     # Last layer

        if i != np.size(W) - 1:
            input = np.r_[Y, 1]
        else:
            input = Y
        net = np.r_[net, 1]
        outputs.append([net, input])
    return outputs


def backward_propagation(outputs, expected, W):
    nextLayerError = 0
    weightUpdates = list()
    for i in reversed(range(len(W))):
        output = outputs[i+1]
        input = outputs[i]
        if i == len(W) - 1:
            error = (expected - output[1]) * activations.tanh_derivative(output[0])

        else:
            error = np.sum(nextLayerError * np.where(np.isnan(W[i + 1]), 0, W[i + 1]), axis=0) * activations.tanh_derivative(output[0])
        nextLayerError = error[0:-1]
        weightUpdates.append(np.dot(nextLayerError[:, np.newaxis], np.transpose(input[1][:, np.newaxis])))
        weightUpdates.reverse()
    return weightUpdates


def update_weights(W, weight_updates, alpha):
    for i in range(0, len(W)):
        W[i] = W[i] + weight_updates[i] * alpha
    return W


X = np.array([0.1, 0.9], dtype='float')
t = np.array([0.5])
alpha = 0.25
W = initialize()
outputs = forward_propagation(X, W)
weight_updates = backward_propagation(outputs, t, W)
new_weights = update_weights(W, weight_updates, alpha)
print(new_weights)
