package com.fsuarez.showcase.supervised;

import com.fsuarez.showcase.Learner;
import com.fsuarez.showcase.util.MatrixUtil;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 * @author fsuarez
 */
public class NormalEquationLinearRegression implements Learner {

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

    @Override
    public RealMatrix predict(RealMatrix X, RealMatrix theta) {
        return X.multiply(theta);
    }

    /**
     * Theta = (X^T * X)^-1 * X^T * y
     *
     * @param X
     * @param y
     * @return theta
     */
    public RealMatrix computeTheta(RealMatrix X, RealVector y) {
        RealMatrix transX = X.transpose();
        RealMatrix inverse = new LUDecomposition(transX.multiply(X)).getSolver().getInverse();

        return inverse.multiply(transX).multiply(MatrixUtils.createColumnRealMatrix(y.toArray()));
    }
}
