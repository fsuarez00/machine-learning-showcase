import numpy as np


def run():
    points = np.loadtxt("lrdata2.txt", delimiter=",")

    # m = number of samples
    # n = number of features
    m, n = points.shape

    rawX = points[:, 0:n-1]

    xNorm = feature_normalize(rawX)

    x = append_bias_term(xNorm)
    print(x)


def feature_normalize(x):
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)

    return (x - mu) / sigma


def append_bias_term(x):
    return np.insert(x, 0, [1], axis=1)


if __name__ == "__main__":
    run()