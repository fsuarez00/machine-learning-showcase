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

        LogisticRegressionCalculator logisticRegressionCalculator = new LogisticRegressionCalculator();
        Fmincg.FmincgReturn result = Fmincg.fMinUnc(logisticRegressionCalculator, X, y, theta, 0.0, iterations);

        RealMatrix input = MatrixUtils.createRowRealMatrix(new double[]{1.0, 45.0, 85.0});
        RealMatrix p = logisticRegressionCalculator.computePrediction(input, result.getTheta());
        RealMatrix prediction = MatrixUtils.createRealMatrix(p.getRowDimension(), 1);
        for(int i = 0; i < p.getRowDimension(); i++)
            if(p.getEntry(i, 0) >= 0.5)
                prediction.setEntry(i, 0, 1.0);
            else
                prediction.setEntry(i, 0, 0.0);

        Assert.assertEquals(1.0, prediction.getData()[0][0], 0.1);
    }

    @Test
    public void gradientDescentRegressionWithRegularizationAndFeatureMappingTest() {
        RealMatrix mData = MatrixUtil.readDataFile("logrdata2.txt");
        RealVector x1 = MatrixUtils.createRealVector(mData.getColumn(0));
        RealVector x2 = MatrixUtils.createRealVector(mData.getColumn(1));

        // create a new a 28 dimensional matrix of which 26 are generated features
        RealMatrix featureMappedX = mData.getSubMatrix(0, mData.getRowDimension() - 1, 0, 1);
        for(int i = 1; i < 7; i++) {
            for(int j = 0; j < i; j++) {
                double[] vector = new double[featureMappedX.getRowDimension()];
                for(int z = 0; z < mData.getRowDimension(); z++)
                    vector[z] = Math.pow(x1.getEntry(z), i - j) * Math.pow(x2.getEntry(z), j);
                featureMappedX = MatrixUtil.appendColumn(featureMappedX, MatrixUtils.createRealVector(vector));
            }
        }

        RealVector y = mData.getColumnVector(mData.getColumnDimension()-1);
        RealMatrix theta = MatrixUtil.getThetaZeros(featureMappedX.getColumnDimension());

        int iterations = 400;

        LogisticRegressionCalculator logisticRegressionCalculator = new LogisticRegressionCalculator();
        Fmincg.FmincgReturn result = Fmincg.fMinUnc(logisticRegressionCalculator, featureMappedX, y, theta, 1.0, iterations);
    }
}
