import numpy as np


def get_output_size(input_size, mask_size, padding, stride, dilation=1):
    (nh, nw, nc) = input_size
    (mh, mw, mc) = mask_size
    nh_out, nw_out = int((nh + 2 * padding - mh) / stride) + 1 - (dilation - 1), int(
        (nw + 2 * padding - mw) / stride) + 1 - (dilation - 1)
    return (nh_out, nw_out, nc)


output_size = get_output_size((11, 15, 6), (3, 3, 6), padding=0, stride=2)
# print(output_size)


def pooling_layer(feature_maps, pool_size, stride, pooling='max'):
    nh, nw = feature_maps[0].shape
    nh_out, nw_out = int((nh - pool_size[0]) / stride) + 1, int((nw - pool_size[1]) / stride) + 1
    out_maps = list()
    for X in feature_maps:
        out = np.zeros((nh_out, nw_out))
        for i in range(nh_out):
            for j in range(nw_out):
                i_start = i * stride
                j_start = j * stride
                i_end = i_start + pool_size[0]
                j_end = j_start + pool_size[1]
                if pooling == 'avg':
                    out[i, j] = np.average(X[i_start:i_end, j_start:j_end])
                else:
                    out[i, j] = np.max(X[i_start:i_end, j_start:j_end])
        out_maps.append(out)
    return(out_maps)


feature_maps = list()
feature_maps.append(np.array([[0.2, 1, 0, 0.4], [-1, 0, -0.1, -0.1], [0.1, 0, -1, -0.5], [0.4, -0.7, -0.5, 1]]))
# output = pooling_layer(feature_maps, (3, 3), stride=1, pooling='max')
# print(output)


def convolution_layer(feature_maps, mask_channels, padding, stride, dilation):
    nh, nw = feature_maps[0].shape
    mh, mw = mask_channels[0].shape
    nh_out, nw_out = int((nh + 2 * padding - mh ) / stride) + 1 - (dilation - 1), int((nw + 2 * padding - mw ) / stride) + 1 - (dilation - 1)
    out = np.zeros((nh_out, nw_out))
    for i in range(nh_out):
        for j in range(nw_out):
            for X, H in zip(feature_maps, mask_channels):
                X = np.pad(X, padding)
                i_start = i * stride
                j_start = j * stride
                i_end = i_start + mh * dilation - dilation + 1
                j_end = j_start + mw * dilation - dilation + 1
                out[i, j] = out[i, j] + np.sum(X[i_start:i_end:dilation, j_start:j_end:dilation] * H)
    return out


feature_maps = list()
feature_maps.append(np.array([[0.2, 1, 0], [-1, 0, -0.1], [0.1, 0, 0.1]]))
feature_maps.append(np.array([[1, 0.5, 0.2], [-1, -0.5, -0.2], [0.1, -0.1, 0]]))

mask_channels = list()
mask_channels.append(np.array([[1, -0.1], [1, -0.1]]))
mask_channels.append(np.array([[0.5, 0.5], [-0.5, -0.5]]))

# output = convolution_layer(feature_maps, mask_channels, padding=0, stride=1, dilation=2)
# print(output)


feature_maps = list()
feature_maps.append(np.array([[0.2, 1, 0], [-1, 0, -0.1], [0.1, 0, 0.1]]))
feature_maps.append(np.array([[1, 0.5, 0.2], [-1, -0.5, -0.2], [0.1, -0.1, 0]]))
feature_maps.append(np.array([[0.5, -0.5, -0.1], [0, -0.4, 0], [0.5, 0.5, 0.2]]))

mask_channels = list()
mask_channels.append(np.array([[1]]))
mask_channels.append(np.array([[-1]]))
mask_channels.append(np.array([[0.5]]))
# output = convolution_layer(feature_maps, mask_channels, padding=0, stride=1, dilation=1)
# print(output)


def batch_normalization(batch, beta, gamma, epsilon):
    mean = np.zeros(batch[0].shape)
    for X in batch:
        mean = mean + X
    mean = mean / len(batch)

    variance = np.zeros(batch[0].shape)
    for X in batch:
        variance = variance + (X - mean) ** 2
    variance = variance / len(batch)

    batch_normalized = list()
    for X in batch:
        batch_normalized.append(beta + gamma * (X - mean) / np.sqrt(variance + epsilon))

    return batch_normalized


X1 = np.array([[1, 0.5, 0.2], [-1, -0.5, -0.2], [0.1, -0.1, 0]])
X2 = np.array([[1, -1, 0.1], [0.5, -0.5, -0.1], [0.2, -0.2, 0]])
X3 = np.array([[0.5, -0.5, -0.1], [0, -0.4, 0], [0.5, 0.5, 0.2]])
X4 = np.array([[0.2, 1, -0.2], [-1, -0.6, -0.1], [0.1, 0, 0.1]])
beta = 0
gamma = 1
epsilon = 0.1
print(np.round(batch_normalization((X1, X2, X3, X4), beta, gamma, epsilon), 2))
