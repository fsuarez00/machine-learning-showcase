package com.fsuarez.ai.calc;

import com.fsuarez.showcase.util.MatrixUtil;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.junit.Assert;
import org.junit.jupiter.api.Test;

/**
 * Created by fsuarez on 5/25/17.
 */
class LinearRegressionCalculatorTest {
    @Test
    void computeCost() {
        LinearRegressionCalculator linearRegressionCalculator = new LinearRegressionCalculator();

        RealMatrix mData = MatrixUtil.readDataFile("lrdata1.txt");
        RealMatrix X = MatrixUtil.appendBiasTermColumnWithOnes(
                mData.getSubMatrix(0, mData.getRowDimension()-1, 0, mData.getColumnDimension()-2));
        RealVector y = mData.getColumnVector(mData.getColumnDimension()-1);
        RealMatrix theta = MatrixUtil.getThetaZeros(X.getColumnDimension());

        double cost = linearRegressionCalculator.computeCost(X, y, theta);
        Assert.assertEquals(32.0727, cost, 0.0001);
    }

    @Test
    void computePrediction() {
        RealMatrix theta = MatrixUtils.createColumnRealMatrix(new double[]{-3.63029143940436, 1.166362350335582});

        RealMatrix input = MatrixUtils.createRowRealMatrix(new double[]{1.0, 3.5});
        RealMatrix p = new LinearRegressionCalculator().computePrediction(input, theta);

        double prediction = p.getData()[0][0] * 10000.0;
        Assert.assertEquals(4519.7678, prediction, 0.0001);
    }

    @Test
    void computeCostDerivative() {
    }

    @Test
    void computeRegularization() {
    }

}