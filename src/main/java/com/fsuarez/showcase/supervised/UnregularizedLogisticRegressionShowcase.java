package com.fsuarez.showcase.supervised;

import java.util.Arrays;
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
public class UnregularizedLogisticRegressionShowcase implements Showcase {

    /**
     * @return algorithm data
     */
    @Override
    public Data run() {
        // Unregularized Logistic Regression
        double[][] matrixData = readDataFile("logrdata1.txt");
        RealMatrix m = MatrixUtils.createRealMatrix(matrixData);
        RealMatrix X = MatrixUtils.createRealMatrix(m.getRowDimension(), m.getColumnDimension()-1);
        X.setColumnMatrix(0, m.getSubMatrix(0, m.getRowDimension()-1, 0, 0));
        X.setColumnMatrix(1, m.getSubMatrix(0, m.getRowDimension()-1, 1, 1));

        RealVector y = m.getColumnVector(m.getColumnDimension()-1);

        X = MatrixUtil.appendBiasTermColumnWithOnes(X);

        // initialize parameters to 0
        RealMatrix theta = MatrixUtil.getThetaZeros(X);

        return new Data(X, y, theta);
    }

    @Override
    public void addToChart(RealMatrix X, RealMatrix theta, RealMatrix jHist, Learner learner) {

    }
}
