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

    @Override
    public RealMatrix computeGradient(RealMatrix X, RealMatrix H, RealVector y) {
        return null;
    }

    public RealMatrix computeRegularization() {
        return null;
    }
}
