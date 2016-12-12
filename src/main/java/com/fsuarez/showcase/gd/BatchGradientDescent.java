package com.fsuarez.showcase.gd;

import java.util.HashMap;
import java.util.Map;
import com.fsuarez.showcase.Learner;
import com.fsuarez.showcase.util.MatrixUtil;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 * @author fsuarez
 */
public class BatchGradientDescent implements GradientDescent {

    private RealMatrix X;       // feature matrix
    private RealVector y;       // label vector
    private RealMatrix theta;   // learned parameters
    private int iterations;     // number of iterations to run gradient descent
    private double alpha;       // learning rate

    private Learner learner;    // learning algorithm

    public BatchGradientDescent(RealMatrix X, RealVector y, RealMatrix theta, int iterations, double alpha, Learner learner) {
        this.X = X;
        this.y = y;
        this.theta = theta;
        this.iterations = iterations;
        this.alpha = alpha;
        this.learner = learner;
    }

    public Map<String, RealMatrix> run() {

        // iteration compute cost
        RealMatrix J = MatrixUtils.createRealMatrix(iterations, 1);//;

        RealMatrix theta = this.theta;
        for(int i = 0; i < iterations; i++) {
            theta = gradientUpdate(theta);
            J.setEntry(i, 0, learner.computeCost(X, y, theta));
        }

        System.out.println("Theta found by gradient descent:");
        System.out.println(MatrixUtil.toString(theta));

        Map<String, RealMatrix> resultMap = new HashMap<>(2);
        resultMap.put("theta", theta);
        resultMap.put("jHist", J);

        return resultMap;
    }

    /**
     * theta_j = theta_j - (alpha / m) * sum((h_theta(x(i)) - y(i)) * x(i)_j)
     * @return updated theta matrix
     */
    private RealMatrix gradientUpdate(RealMatrix theta) {
        // number of training examples
        int m = y.getDimension();

        RealMatrix h = learner.predict(X, theta);

        RealMatrix error = h.subtract(MatrixUtils.createColumnRealMatrix(y.toArray()));

        RealMatrix thetaChange = X.transpose().multiply(error).scalarMultiply(alpha / (double)m);

        return theta.subtract(thetaChange);
    }

}
