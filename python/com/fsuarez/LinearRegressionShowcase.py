import numpy as np
import util


def run():
    points = np.loadtxt("lrdata2.txt", delimiter=",")

    # m = number of samples
    # n = number of features
    m, n = points.shape
    n -= 1

    raw_x = points[:, 0:n]

    x_norm = util.feature_normalize(raw_x, raw_x)

    x = util.append_bias_term(x_norm)
    y = points[:, n]
    y = np.reshape(y, (m, 1))
    theta = np.zeros((x.shape[1], 1))

    # batch gradient descent
    alpha = 0.01
    iterations = 400
    for i in range(0, iterations):
        h = x.dot(theta)
        error = h - y
        theta_change = (alpha / m) * (x.T.dot(error))
        theta -= theta_change

    input = np.array([[1650.0, 3.0]])
    input_norm = util.feature_normalize(raw_x, input)
    input_norm = util.append_bias_term(input_norm)

    prediction = input_norm.dot(theta)

    print(prediction)


if __name__ == "__main__":
    run()