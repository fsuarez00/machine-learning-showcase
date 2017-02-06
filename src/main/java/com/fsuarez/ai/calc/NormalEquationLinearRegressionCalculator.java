package com.fsuarez.ai.calc;

import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 * @author fsuarez
 */
public class NormalEquationLinearRegressionCalculator implements Calculator {

    /**
     * No need to compute cost for minimization.
     *
     * @param X
     * @param y
     * @param theta
     * @return
     */
    @Override
    public double computeCost(RealMatrix X, RealVector y, RealMatrix theta) {
        return 0;
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
        return null;
    }

    @Override
    public RealMatrix computeGradient(RealMatrix X, RealMatrix H, RealVector y) {
        RealMatrix transX = X.transpose();
        RealMatrix inverse = new LUDecomposition(transX.multiply(X)).getSolver().getInverse();

        return inverse.multiply(transX).multiply(MatrixUtils.createColumnRealMatrix(y.toArray()));
    }

    @Override
    public double computeRegularization(RealMatrix theta, double lambda, int m) {
        return null;
    }
}
