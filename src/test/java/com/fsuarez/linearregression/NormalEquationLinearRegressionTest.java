package com.fsuarez.linearregression;

import com.fsuarez.ai.calc.NormalEquationLinearRegressionCalculator;
import com.fsuarez.showcase.util.MatrixUtil;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.junit.Assert;
import org.junit.jupiter.api.Test;

/**
 * @author fsuarez
 */
public class NormalEquationLinearRegressionTest {

    @Test
    public void learningTest() {
        RealMatrix mData = MatrixUtil.readDataFile("lrdata2.txt");
        RealMatrix X = MatrixUtil.appendBiasTermColumnWithOnes(
                mData.getSubMatrix(0, mData.getRowDimension()-1, 0, mData.getColumnDimension()-2));
        RealVector y = mData.getColumnVector(mData.getColumnDimension()-1);

        RealMatrix learnedTheta = NormalEquationLinearRegressionCalculator.computeTheta(X, y);

        RealMatrix input = MatrixUtils.createRowRealMatrix(new double[]{1.0, 1650.0, 3.0});
        RealMatrix p = NormalEquationLinearRegressionCalculator.computePrediction(input, learnedTheta);

        double prediction = p.getData()[0][0];
        Assert.assertEquals(293081.4643, prediction, 0.0001);
    }
}
