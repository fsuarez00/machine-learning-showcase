package com.fsuarez.linearregression;

import java.util.Arrays;
import com.fsuarez.ai.calc.LinearRegressionCalculator;
import com.fsuarez.showcase.gd.BatchGradientDescent;
import com.fsuarez.showcase.util.MatrixUtil;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.junit.Assert;
import org.junit.jupiter.api.Test;

/**
 * @author fsuarez
 */
public class MultivariateLinearRegressionTest {

    @Test
    public void gradientDescentPrediction() {
        LinearRegressionCalculator linearRegressionCalculator = new LinearRegressionCalculator();

        RealMatrix mData = MatrixUtil.readDataFile("lrdata2.txt");
        RealMatrix rawX = mData.getSubMatrix(0, mData.getRowDimension()-1, 0, mData.getColumnDimension()-2);

        // feature normalization
        //double[][] normalizedX = featureNormalize(rawX);
        // number of training examples
        int m = rawX.getRowDimension();
        // number of features
        int n = rawX.getColumnDimension();

        // calculate means and standard deviations
        double[] featureMeans = new double[n];
        double[] featureStds = new double[n];
        for(int i = 0; i < n; i++) {
            SummaryStatistics stats = new SummaryStatistics();
            double[] col = rawX.getColumn(i);
            for(double value : col)
                stats.addValue(value);
            // store mean of feature i
            featureMeans[i] = stats.getMean();
            featureStds[i] = stats.getStandardDeviation();
        }

        double[] ones = new double[m];
        Arrays.fill(ones, 1.0);
        RealMatrix mu = MatrixUtils.createRealMatrix(m, n);
        for(int i = 0; i < n; i++)
            mu.setColumnMatrix(i, MatrixUtils.createColumnRealMatrix(ones).scalarMultiply(featureMeans[i]));

        double[][] temp = rawX.subtract(mu).getData();
        for(int i = 0; i < m; i++)
            for(int j = 0; j < n; j++)
                temp[i][j] /= featureStds[j];

        RealMatrix xNorm = MatrixUtils.createRealMatrix(temp);
        RealMatrix xMatrix = MatrixUtil.appendBiasTermColumnWithOnes(xNorm);
        RealVector y = mData.getColumnVector(mData.getColumnDimension()-1);
        RealMatrix theta = MatrixUtil.getThetaZeros(xMatrix.getColumnDimension());

        int iterations = 2000;
        double alpha = 1;

        BatchGradientDescent batchGradientDescent = new BatchGradientDescent(xMatrix, y, theta, iterations, alpha, linearRegressionCalculator);
        RealMatrix learnedTheta = batchGradientDescent.run();

        RealMatrix input = MatrixUtils.createRowRealMatrix(new double[]{1650.0, 3.0});
        temp = input.subtract(mu.getSubMatrix(0, 0, 0, input.getColumnDimension()-1)).getData();
        for(int i = 0; i < input.getRowDimension(); i++)
            for(int j = 0; j < input.getColumnDimension(); j++)
                temp[i][j] /= featureStds[j];

        RealMatrix p = linearRegressionCalculator.computePrediction(MatrixUtil.appendBiasTermColumnWithOnes(MatrixUtils.createRealMatrix(temp)), learnedTheta);

        double prediction = p.getData()[0][0];
        Assert.assertEquals(293081.4643, prediction, 0.0001);
    }
}
