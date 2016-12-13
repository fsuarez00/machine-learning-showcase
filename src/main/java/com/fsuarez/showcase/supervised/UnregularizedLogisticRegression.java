package com.fsuarez.showcase.supervised;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.util.FastMath;

/**
 * @author fsuarez
 */
public class UnregularizedLogisticRegression implements LogisticRegression {

    /**
     *  J(theta) = 1/m * sum(-y(i)*log(h_theta(x(i))) - (1 - y(i)) * log(1 - h_theta(x(i))))
     *
     *  Vectorized:
     *  J(theta) = (-y^T * log(h) - (1 - y)^T * log(1 - h)) / m
     *
     * @param X
     * @param y
     * @param theta
     * @return J
     */
    @Override
    public double computeCost(RealMatrix X, RealVector y, RealMatrix theta) {
        // number of training examples
        int m = y.getDimension();

        RealMatrix H = sigmoid(X.multiply(theta));

        RealMatrix negYMatrix = MatrixUtils.createRealMatrix(y.getDimension(), 1);
        for(int i = 0; i < negYMatrix.getRowDimension(); i++)
            negYMatrix.setEntry(i, 0, -y.getEntry(i));

        // (1 - y)^T
        RealMatrix oneMinusYTranspose = negYMatrix.scalarAdd(1.0).transpose();

        // calculate log(h)
        RealMatrix logH = MatrixUtils.createRealMatrix(H.getRowDimension(), H.getColumnDimension());
        for(int i = 0; i < H.getRowDimension(); i++)
            logH.setEntry(i, 0, FastMath.log(H.getEntry(i, 0)));

        // calculate log(1 - h)
        RealMatrix logOneMinusH = MatrixUtils.createRealMatrix(H.getRowDimension(), H.getColumnDimension());
        for(int i = 0; i < H.getRowDimension(); i++)
            logOneMinusH.setEntry(i, 0, 1.0 - H.getEntry(i, 0));

        RealMatrix J = negYMatrix.transpose().multiply(logH).subtract(oneMinusYTranspose.multiply(logOneMinusH)).scalarMultiply(1.0/m);

        return J.getEntry(0, 0);
    }

    /**
     * g(z) = 1 / (1 + e^-z)
     * h_theta(x) = g(theta^T * X)
     *
     * @param X
     * @param theta
     * @return h
     */
    @Override
    public RealMatrix predict(RealMatrix X, RealMatrix theta) {
        RealMatrix H = sigmoid(X.multiply(theta));
        RealMatrix P = MatrixUtils.createRealMatrix(X.getRowDimension(), 0);
        for(int i = 0; i < X.getRowDimension(); i++)
            if(H.getEntry(i, 0) >= 0.5)
                P.setEntry(i, 0, 1.0);
            else
                P.setEntry(i, 0, 0.0);

        return P;
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
    public RealMatrix computeGradient(RealMatrix X, RealMatrix H, RealVector y) {
        // number of training examples
        int m = y.getDimension();

        RealMatrix yMatrix = MatrixUtils.createColumnRealMatrix(y.toArray());
        return X.transpose().multiply(H.subtract(yMatrix)).scalarMultiply(1.0 / m);
    }

    private RealMatrix sigmoid(RealMatrix Z) {
        RealMatrix G = MatrixUtils.createRealMatrix(Z.getRowDimension(), Z.getColumnDimension());
        for(int i = 0; i < Z.getRowDimension(); i++)
            for(int j = 0; j < Z.getColumnDimension(); j++)
                G.setEntry(i, j, sigmoid(Z.getEntry(i, j)));
        return G;
    }

    private double sigmoid(double value) {
        return 1.0 / (1.0 + FastMath.exp(-value));
    }
}
