package com.fsuarez.showcase.gd;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 * @author fsuarez
 */
public class BatchGradientDescent {

    private RealMatrix X;       // feature matrix
    private RealVector y;       // label vector
    private RealMatrix theta;   // learned parameters
    private int iterations;     // number of iterations to run gradient descent
    private double alpha;       // learning rate

    public BatchGradientDescent(RealMatrix X, RealVector y, RealMatrix theta, int iterations, double alpha) {
        this.X = X;
        this.y = y;
        this.theta = theta;
        this.iterations = iterations;
        this.alpha = alpha;
    }

    public RealMatrix run() {
        RealMatrix theta = this.theta;
        for(int i = 0; i < iterations; i++)
            theta = gradientUpdate(theta);

        System.out.println("Theta found by gradient descent: " + theta.getRow(0)[0] + ", " + theta.getRow(1)[0]);

        return theta;
    }

    /**
     * theta_j = theta_j - (alpha / m) * sum((h_theta(x(i)) - y(i)) * x(i)_j)
     * @return updated theta matrix
     */
    private RealMatrix gradientUpdate(RealMatrix theta) {
        // number of training examples
        int m = y.getDimension();

        RealMatrix h = X.multiply(theta);

        RealMatrix error = h.subtract(MatrixUtils.createColumnRealMatrix(y.toArray()));

        RealMatrix thetaChange = X.transpose().multiply(error).scalarMultiply(alpha / (double)m);

        return theta.subtract(thetaChange);
    }

}
