package com.fsuarez.linearregression;

import com.fsuarez.ai.calc.Fmincg;
import com.fsuarez.ai.calc.LogisticRegressionCalculator;
import com.fsuarez.showcase.gd.BatchGradientDescent;
import com.fsuarez.showcase.util.MatrixUtil;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.junit.Assert;
import org.junit.Test;

/**
 * @author fsuarez
 */
public class LogisticRegressionTest {

    @Test
    public void gradientDescentRegressionTest() {
        LogisticRegressionCalculator logisticRegressionCalculator = new LogisticRegressionCalculator();

        RealMatrix mData = MatrixUtil.readDataFile("logrdata1.txt");
        RealMatrix X = MatrixUtil.appendBiasTermColumnWithOnes(
                mData.getSubMatrix(0, mData.getRowDimension()-1, 0, mData.getColumnDimension()-2));
        RealVector y = mData.getColumnVector(mData.getColumnDimension()-1);
        RealMatrix theta = MatrixUtil.getThetaZeros(X.getColumnDimension());

        int iterations = 1500;
        double alpha = 0.01;

        BatchGradientDescent batchGradientDescent = new BatchGradientDescent(X, y, theta, iterations, alpha, logisticRegressionCalculator);
        RealMatrix learnedTheta = batchGradientDescent.run();

        RealMatrix input = MatrixUtils.createRowRealMatrix(new double[]{1.0, 45.0, 85.0});
        RealMatrix p = logisticRegressionCalculator.computePrediction(input, learnedTheta);
        RealMatrix prediction = MatrixUtils.createRealMatrix(p.getRowDimension(), 1);
        for(int i = 0; i < p.getRowDimension(); i++)
            if(p.getEntry(i, 0) >= 0.5)
                prediction.setEntry(i, 0, 1.0);
            else
                prediction.setEntry(i, 0, 0.0);

        Assert.assertEquals(1.0, prediction.getData()[0][0], 0.1);
    }

    @Test
    public void fMinUncRegressionTest() {
        RealMatrix mData = MatrixUtil.readDataFile("logrdata1.txt");
        RealMatrix X = MatrixUtil.appendBiasTermColumnWithOnes(
                mData.getSubMatrix(0, mData.getRowDimension()-1, 0, mData.getColumnDimension()-2));
        RealVector y = mData.getColumnVector(mData.getColumnDimension()-1);
        RealMatrix theta = MatrixUtil.getThetaZeros(X.getColumnDimension());

        int iterations = 400;

        Fmincg.fMinUnc(new LogisticRegressionCalculator(), X, y, theta, 0.0, iterations);
    }
}
