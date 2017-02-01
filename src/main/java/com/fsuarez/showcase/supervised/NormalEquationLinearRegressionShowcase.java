package com.fsuarez.showcase.supervised;

import com.fsuarez.showcase.Data;
import com.fsuarez.showcase.Learner;
import com.fsuarez.showcase.Showcase;
import com.fsuarez.showcase.util.MatrixUtil;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 * @author fsuarez
 */
public class NormalEquationLinearRegressionShowcase implements Showcase {

    /**
     * @return algorithm data
     */
    @Override
    public Data run() {
        // Linear Regression using normal equations
        // no feature scaling is required
        double[][] matrixData = readDataFile("lrdata2.txt");
        RealMatrix m = MatrixUtils.createRealMatrix(matrixData);

        RealVector y = m.getColumnVector(m.getColumnDimension()-1);

        RealMatrix X = MatrixUtil.appendBiasTermColumnWithOnes(m.getSubMatrix(0, m.getRowDimension()-1, 0, m.getColumnDimension()-2));

        NormalEquationLinearRegression learner = new NormalEquationLinearRegression();
        RealMatrix theta = learner.computeTheta(X, y);

        System.out.println("Theta found by normal equation:");
        System.out.println(MatrixUtil.toString(theta));

        return new Data(X, y, theta);
    }

    @Override
    public void addToChart(RealMatrix X, RealMatrix theta, RealMatrix jHist, Learner learner) {

    }
}
