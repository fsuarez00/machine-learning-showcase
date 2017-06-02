package com.fsuarez.linearregression;

import com.fsuarez.ai.calc.LinearRegressionCalculator;
import com.fsuarez.showcase.gd.BatchGradientDescent;
import com.fsuarez.showcase.util.MatrixUtil;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.junit.Assert;
import org.junit.jupiter.api.Test;

public class UnivariateLinearRegressionTest {

    @Test
    public void gradientDescentRegressionTest() {
        LinearRegressionCalculator linearRegressionCalculator = new LinearRegressionCalculator();

        RealMatrix mData = MatrixUtil.readDataFile("lrdata1.txt");
        RealMatrix X = MatrixUtil.appendBiasTermColumnWithOnes(
                mData.getSubMatrix(0, mData.getRowDimension()-1, 0, mData.getColumnDimension()-2));
        RealVector y = mData.getColumnVector(mData.getColumnDimension()-1);
        RealMatrix theta = MatrixUtil.getThetaZeros(X.getColumnDimension());

        int iterations = 1500;
        double alpha = 0.01;

        BatchGradientDescent batchGradientDescent = new BatchGradientDescent(X, y, theta, iterations, alpha, linearRegressionCalculator);
        RealMatrix learnedTheta = batchGradientDescent.run();

        RealMatrix input = MatrixUtils.createRowRealMatrix(new double[]{1.0, 3.5});
        RealMatrix p = linearRegressionCalculator.computePrediction(input, learnedTheta);

        double prediction1 = p.getData()[0][0] * 10000.0;
        Assert.assertEquals(4519.7678, prediction1, 0.0001);

        input = MatrixUtils.createRowRealMatrix(new double[]{1.0, 7.0});
        p = linearRegressionCalculator.computePrediction(input, learnedTheta);

        double prediction2 = p.getData()[0][0] * 10000.0;
        Assert.assertEquals(45342.45012, prediction2, 0.0001);
    }
}