import numpy as np


def map_features(x1, x2):
    degree = 6
    out = None
    for i in range(1, degree+1):
        for j in range(0, i+1):
            x_1 = np.reshape(np.power(x1, i - j), (47, 1))
            x_2 = np.reshape(np.power(x2, j), (47, 1))
            if i is 1 and j is 0:
                out = x_1 * x_2
            else:
                out = np.append(out, x_1 * x_2, axis=1)

    return out


def append_bias_term(x):
    return np.insert(x, 0, [1], axis=1)


def feature_normalize(source, x):
    mu = np.mean(source, axis=0)
    sigma = np.std(source, axis=0)

    return (x - mu) / sigma
