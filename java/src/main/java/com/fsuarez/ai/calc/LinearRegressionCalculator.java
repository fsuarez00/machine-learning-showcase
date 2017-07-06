package com.fsuarez.ai.calc;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 * @author fsuarez
 */
public class LinearRegressionCalculator implements Calculator {

    /**
     * J(theta) = 1/2m * sum(h_theta(x(i)) - y(i))^2
     *
     * @param X feature matrix
     * @param y labels vector
     * @param theta parameters matrix
     * @return J
     */
    @Override
    public double computeCost(RealMatrix X, RealVector y, RealMatrix theta) {
        // number of training examples
        int m = y.getDimension();

        RealMatrix h = computePrediction(X, theta);

        // get squared error and sum
        RealMatrix error = h.subtract(MatrixUtils.createColumnRealMatrix(y.toArray()));
        double[][] sqrErrorArray = error.getData();
        double sum = 0.0;
        for(int i = 0; i < error.getRowDimension(); i++) {
            sqrErrorArray[i][0] *= sqrErrorArray[i][0];
            sum += sqrErrorArray[i][0];
        }

        double J = sum / (2.0 * (double)m);

        return J;
    }

    /**
     * h_theta(x) = theta^T * X = theta(0) + theta(1)*X(1)
     *
     * @param X
     * @param theta
     * @return h
     */
    @Override
    public RealMatrix computePrediction(RealMatrix X, RealMatrix theta) {
        return X.multiply(theta);
    }

    /**
     *  Derivative of J(theta) w.r.t. theta_j:
     *  1/m * sum(h_theta(x(i)) - y(i)) * x_j(i))
     *
     *  Vectorized:
     *  X^T * (h - y) / m
     *
     * @param X
     * @param H
     * @param y
     * @return gradient of the cost w.r.t. the parameters
     */
    @Override
    public RealMatrix computeCostDerivative(RealMatrix X, RealMatrix H, RealVector y) {
        // number of training examples
        int m = y.getDimension();

        RealMatrix yMatrix = MatrixUtils.createColumnRealMatrix(y.toArray());
        return X.transpose().multiply(H.subtract(yMatrix)).scalarMultiply(1.0 / m);
    }

    @Override
    public RealMatrix computeCostDerivativeRegularization(RealMatrix theta, double lamda, int m) {
        return null;
    }

    @Override
    public double computeCostRegularization(RealMatrix theta, double lambda, int m) {
        return 0.0;
    }
}
