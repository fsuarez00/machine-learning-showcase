import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.io as io
import util

def run():
    unregularized()
    regularized()
    one_vs_all()


def one_vs_all():
    data = io.loadmat("textimgdata1.mat")
    x = data['X']
    y = data['y']

    num_labels = 10

    m, n = x.shape

    lmd = 0.1

    # theta matrix containing theta values for each class
    theta = np.zeros((num_labels, n + 1))
    # append bias term
    x = util.append_bias_term(x)
    # train multiple logistic regression classifiers
    for c in range(1, num_labels + 1):  # labels are 1-indexed instead of 0-indexed
        # convert y to a binary value for each classifier
        y_i = np.array([1 if label == c else 0 for label in y])
        y_i = np.reshape(y_i, (m ,1))

        fmin = opt.minimize(fun=cost_reg, x0=theta[c-1], args=(x, y_i, lmd), method='TNC', jac=gradient_reg)

        theta[c-1] = fmin.x

    prediction = predict_one_vs_all(theta, x)

    positive = [1 if a == b else 0 for a, b in zip(prediction, y)]
    accuracy = sum(map(int, positive)) / float(len(positive))
    print("Training Accuracy: ", accuracy * 100, "%")


def regularized():
    points = np.loadtxt("logrdata2.txt", delimiter=",")
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

    plt.scatter(x1, y1, marker='x', color='black', label='y = 1')
    plt.scatter(x2, y2, marker='o', color='y', label='y = 0')
    plt.xlabel('Monochip Test 1')
    plt.ylabel('Monochip Test 2')
    plt.legend(loc='best')
    plt.show()

    x = util.map_features(raw_x[:, 0], raw_x[:, 1])
    x = util.append_bias_term(x)
    # m = number of samples
    # n = number of features
    m, n = x.shape
    theta = np.zeros((n, 1))
    lmd = 1  # lambda

    j = cost_reg(theta, x, y, lmd)
    grad = gradient_reg(theta, x, y, lmd)

    print("Cost of initial theta (zeros): ", j)
    print("Gradient at initial theta (zeros): ", grad)

    result = opt.fmin_tnc(func=cost_reg, x0=theta, fprime=gradient_reg, args=(x, y, lmd))
    theta = np.reshape(result[0], (n, 1))

    j = cost_reg(theta, x, y, lmd)
    grad = gradient_reg(theta, x, y, lmd)

    print("Cost of final theta: ", j)
    print("Gradient at final theta: ", grad)

    # compute accuracy of training set
    p = predict(theta, x)
    positive = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for a, b in zip(p, y)]
    accuracy = sum(map(int, positive)) % len(positive)
    print("Training Accuracy: ", accuracy, "%")


def unregularized():
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

    j = cost(theta, x, y)
    grad = gradient(theta, x, y)

    print("Cost of initial theta (zeros): ", j)
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
    theta = np.reshape(result[0], (n + 1, 1))

    j = cost(theta, x, y)
    grad = gradient(theta, x, y)

    print("Cost of final theta: ", j)
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

    prob = sigmoid(np.array([1, 45, 85]).dot(theta))
    print("For a student with scores 45 and 85, we predict an admission probability of ", prob[0])

    # compute accuracy of training set
    p = predict(theta, x)
    positive = [1 if ((a == 1 and b == 1) or (a ==0 and b == 0)) else 0 for a, b in zip(p, y)]
    accuracy = sum(map(int, positive)) % len(positive)
    print("Training Accuracy: ", accuracy, "%")


# computes the predictions for X using a threshold at 0.5
def predict(theta, x):
    h = sigmoid(x.dot(theta))

    return [1 if x >= 0.5 else 0 for x in h]


def predict_one_vs_all(theta, x):
    h = sigmoid(x.dot(theta.T))

    # get indices with max probability
    h_argmax = np.argmax(h, axis=1)

    # add 1 due to zero indexing
    h_argmax += 1

    return h_argmax


# Computes the cost of using theta as the parameter
def cost(theta, x, y):
    m, n = x.shape

    h = sigmoid(x.dot(theta))
    h = np.reshape(h, (m, 1))

    j = (-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))) / m

    return j


def cost_reg(theta, x, y, lmd):
    m, n = x.shape

    h = sigmoid(x.dot(theta))
    h = np.reshape(h, (m, 1))

    cost = (-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))) / m
    reg = (lmd * sum(map(float, np.power(theta[1:], 2)))) / (2 * m)
    j = cost + reg

    return j


# Computes the gradient of the cost w.r.t. to the parameters
def gradient(theta, x, y):
    m, n = x.shape

    h = sigmoid(x.dot(theta))
    h = np.reshape(h, (m, 1))

    error = h - y

    grad = (x.T.dot(error)) / m

    return np.squeeze(np.asarray(grad))


def gradient_reg(theta, x, y, lmb):
    m, n = x.shape

    h = sigmoid(x.dot(theta))
    h = np.reshape(h, (m, 1))

    error = h - y

    grad = (x.T.dot(error)) / m
    for i in range(1, len(grad)):
        grad[i] = grad[i] + (lmb * theta[i] / m)

    return np.squeeze(np.asarray(grad))


# Compute sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


if __name__ == "__main__":
    run()
