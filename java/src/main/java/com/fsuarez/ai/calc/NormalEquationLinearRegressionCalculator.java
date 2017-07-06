package com.fsuarez.ai.calc;

import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 * @author fsuarez
 */
public class NormalEquationLinearRegressionCalculator {

    private NormalEquationLinearRegressionCalculator(){}

    /**
     * h_theta(x) = theta^T * X = theta(0) + theta(1)*X(1)
     *
     * @param X
     * @param theta
     * @return h
     */
    public static RealMatrix computePrediction(RealMatrix X, RealMatrix theta) {
        return X.multiply(theta);
    }

    /**
     * Theta = (X^T * X)^-1 * X^T * y
     *
     * @param X
     * @param y
     * @return theta
     */
    public static RealMatrix computeTheta(RealMatrix X, RealVector y) {
        RealMatrix transX = X.transpose();
        RealMatrix inverse = new LUDecomposition(transX.multiply(X)).getSolver().getInverse();

        return inverse.multiply(transX).multiply(MatrixUtils.createColumnRealMatrix(y.toArray()));
    }
}
