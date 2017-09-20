import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import util

def run():
    points = np.loadtxt("logrdata1.txt", delimiter=",")
    # m = number of samples
    # n = number of features
    m, n = points.shape
    n -= 1

    raw_x = points[:, 0:n]

    y = points[:, n]
    y = np.reshape(y, (m, 1))

    positive = [i for i, j in enumerate(y) if j == 1]
    negative = [i for i, j in enumerate(y) if j == 0]

    x1 = raw_x[positive, 0]
    y1 = raw_x[positive, 1]
    x2 = raw_x[negative, 0]
    y2 = raw_x[negative, 1]

    plt.scatter(x1, y1, marker='x', color='black', label='Admitted')
    plt.scatter(x2, y2, marker='o', color='y', label='Not Admitted')
    plt.xlabel('Exam 1 Score')
    plt.ylabel('Exam 2 Score')
    plt.legend(loc='best')
    plt.show()

    x = util.append_bias_term(raw_x)
    theta = np.zeros((n + 1, 1))

    J = cost(theta, x, y)
    grad = gradient(theta, x, y)

    print("Cost of initial theta (zeros): ", J)
    print("Gradient at initial theta (zeros): ", grad)

    # batch gradient descent
    # alpha = 0.01
    # iterations = 400
    # for i in range(0, iterations):
    #     h = sigmoid(x.dot(theta))
    #     error = h - y
    #     theta_change = (alpha / m) * (x.T.dot(error))
    #     theta -= theta_change

    # alternative to gradient descent which is optimized minimization given a cost function and a gradient function
    result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(x, y))
    theta = np.reshape(result[0], (n+1, 1))

    J = cost(theta, x, y)
    grad = gradient(theta, x, y)

    print("Cost of final theta: ", J)
    print("Gradient at final theta: ", grad)

    # only need 2 points to define a line, so choose 2 endpoints
    lx = np.array([np.amin(x[:, 1]) - 2, np.amax(x[:, 2]) + 2])
    # calculate the decision boundary line
    ly = (-1 / theta[2, 0]) * (theta[1, 0] * lx + theta[0, 0])

    plt.scatter(x1, y1, marker='x', color='black', label='Admitted')
    plt.scatter(x2, y2, marker='o', color='y', label='Not Admitted')
    plt.xlabel('Exam 1 Score')
    plt.ylabel('Exam 2 Score')
    plt.plot(lx, ly, label='Decision Boundary')
    plt.legend(loc='best')
    plt.show()


# Computes the cost of using theta as the parameter
def cost(theta, x, y):
    m, n = x.shape

    h = sigmoid(x.dot(theta))
    h = np.reshape(h, (m, 1))

    j = (-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))) / m

    return j


# Computes the gradient of the cost w.r.t. to the parameters
def gradient(theta, x, y):
    m, n = x.shape

    h = sigmoid(x.dot(theta))
    h = np.reshape(h, (m, 1))

    error = h - y

    grad = (x.T.dot(error)) / m

    return np.squeeze(np.asarray(grad))


# Compute sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


if __name__ == "__main__":
    run()
