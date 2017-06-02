package com.fsuarez.ai.calc;

import com.fsuarez.showcase.util.MatrixUtil;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.junit.Assert;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * @author fsuarez
 */
class NormalEquationLinearRegressionCalculatorTest {

    @Test
    void computePrediction() {
        RealMatrix theta = MatrixUtils.createColumnRealMatrix(new double[]{89597.90954279636, 139.2106740176255, -8738.019112327795});
        RealMatrix input = MatrixUtils.createRowRealMatrix(new double[]{1650.0, 3.0});
        RealMatrix p = NormalEquationLinearRegressionCalculator.computePrediction(MatrixUtil.appendBiasTermColumnWithOnes(input), theta);

        double prediction = p.getData()[0][0];
        Assert.assertEquals(293081.4643, prediction, 0.0001);
    }

    @Test
    void computeTheta() {
        RealMatrix mData = MatrixUtil.readDataFile("lrdata2.txt");
        RealMatrix X = MatrixUtil.appendBiasTermColumnWithOnes(
                mData.getSubMatrix(0, mData.getRowDimension()-1, 0, mData.getColumnDimension()-2));
        RealVector y = mData.getColumnVector(mData.getColumnDimension()-1);

        RealMatrix learnedTheta = NormalEquationLinearRegressionCalculator.computeTheta(X, y);

        RealMatrix expected = MatrixUtils.createColumnRealMatrix(new double[]{89597.90954279636, 139.2106740176255, -8738.019112327795});
        Assert.assertEquals(expected, learnedTheta);
    }

}