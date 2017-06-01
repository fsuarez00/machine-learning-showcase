package com.fsuarez.ai.calc;

import com.fsuarez.showcase.util.MatrixUtil;
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
        RealMatrix theta = MatrixUtil.getThetaZeros(X);

        double cost = linearRegressionCalculator.computeCost(X, y, theta);

        Assert.assertEquals(cost,32.0727, 0.0001);
    }

    @Test
    void computePrediction() {
    }

    @Test
    void computeCostDerivative() {
    }

    @Test
    void computeRegularization() {
    }

}