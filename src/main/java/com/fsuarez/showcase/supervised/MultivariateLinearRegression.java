package com.fsuarez.showcase.supervised;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 * @author fsuarez
 */
public class MultivariateLinearRegression implements LinearRegression {

    /**
     * h_theta(x) = transpose(theta) * X = theta(0) + theta(1)*X(1)
     * @param X
     * @param theta
     * @return prediction
     */
    @Override
    public RealMatrix predict(RealMatrix X, RealMatrix theta) {
        return null;
    }

    /**
     * h_theta(x) = transpose(theta) * X = theta(0) + theta(1)*X(1)
     * J(theta) = 1/2m * sum(h_theta(x(i)) - y(i))^2
     * @param X feature matrix
     * @param y labels vector
     * @param theta parameters matrix
     * @return J
     */
    @Override
    public double computeCost(RealMatrix X, RealVector y, RealMatrix theta) {
        return 0;
    }
}
