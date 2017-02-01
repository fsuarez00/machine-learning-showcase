package com.fsuarez.showcase.gd;

import com.fsuarez.ai.calc.Calculator;
import com.fsuarez.showcase.util.MatrixUtil;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author fsuarez
 */
public class BatchGradientDescent implements GradientDescent {

    private static final Logger LOGGER = LoggerFactory.getLogger(BatchGradientDescent.class);

    private RealMatrix X;       // feature matrix
    private RealVector y;       // label vector
    private RealMatrix theta;   // learned parameters
    private int iterations;     // number of iterations to run gradient descent
    private double alpha;       // learning rate

    private Calculator calculator;    // learning algorithm

    public BatchGradientDescent(RealMatrix X, RealVector y, RealMatrix theta, int iterations, double alpha, Calculator calculator) {
        this.X = X;
        this.y = y;
        this.theta = theta;
        this.iterations = iterations;
        this.alpha = alpha;
        this.calculator = calculator;
    }

    public RealMatrix run() {
        RealMatrix theta = this.theta;
        for(int i = 0; i < iterations; i++)
            theta = descend(theta);

        LOGGER.info("Theta found by gradient descent:\n\n{}", MatrixUtil.toString(theta));

        return theta;
    }

    public RealMatrix descend(RealMatrix theta) {
        return gradientUpdate(theta);
    }

    /**
     * theta_j = theta_j - (alpha / m) * sum((h_theta(x(i)) - y(i)) * x(i)_j)
     * @return updated theta matrix
     */
    private RealMatrix gradientUpdate(RealMatrix theta) {
        // number of training examples
        int m = y.getDimension();

        RealMatrix h = calculator.computePrediction(X, theta);

        RealMatrix error = h.subtract(MatrixUtils.createColumnRealMatrix(y.toArray()));

        RealMatrix thetaChange = X.transpose().multiply(error).scalarMultiply(alpha / (double)m);

        return theta.subtract(thetaChange);
    }

}
