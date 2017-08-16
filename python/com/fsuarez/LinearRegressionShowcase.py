import numpy as np
import util


def run():
    points = np.loadtxt("lrdata2.txt", delimiter=",")

    # m = number of samples
    # n = number of features
    m, n = points.shape
    n -= 1

    rawX = points[:, 0:n]

    xNorm = util.feature_normalize(rawX)

    x = util.append_bias_term(xNorm)
    y = points[:, n]
    theta = np.zeros((1, x.shape[1]))

    print(x)


if __name__ == "__main__":
    run()