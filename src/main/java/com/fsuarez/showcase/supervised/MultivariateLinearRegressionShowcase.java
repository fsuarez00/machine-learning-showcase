package com.fsuarez.showcase.supervised;

import java.util.Arrays;
import com.fsuarez.showcase.Data;
import com.fsuarez.showcase.Learner;
import com.fsuarez.showcase.Showcase;
import com.fsuarez.showcase.chart.LineChart;
import com.fsuarez.showcase.util.MatrixUtil;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.jfree.ui.RefineryUtilities;

/**
 * @author fsuarez
 */
public class MultivariateLinearRegressionShowcase implements Showcase {

    /**
     * @return algorithm data
     */
    @Override
    public Data run() {
        // Multivariate Linear Regression
        double[][] matrixData = readDataFile("lrdata2.txt");
        RealMatrix m = MatrixUtils.createRealMatrix(matrixData);
        RealMatrix X = MatrixUtils.createRealMatrix(m.getRowDimension(), m.getColumnDimension()-1);
        X.setColumnMatrix(0, m.getSubMatrix(0, m.getRowDimension()-1, 0, 0));
        X.setColumnMatrix(1, m.getSubMatrix(0, m.getRowDimension()-1, 1, 1));

        RealVector y = m.getColumnVector(m.getColumnDimension()-1);

        System.out.println("Feature Normalization...");
        RealMatrix xNorm = featureNormalize(X);

        X = MatrixUtil.appendBiasTermColumnWithOnes(xNorm);

        // initialize parameters to 0
        RealMatrix theta = MatrixUtil.getThetaZeros(X);

        return new Data(X, y, theta);
    }

    @Override
    public void addToChart(RealMatrix X, RealMatrix theta, RealMatrix jHist, Learner learner) {
        LineChart chart = new LineChart("Learning Curve");
        chart.createChart(jHist);
        chart.pack();
        RefineryUtilities.centerFrameOnScreen(chart);
        chart.setVisible(true);
    }

    private RealMatrix featureNormalize(RealMatrix X) {
        // number of training examples
        int m = X.getRowDimension();
        // number of features
        int n = X.getColumnDimension();

        // calculate means and standard deviations
        double[] featureMeans = new double[n];
        double[] featureStds = new double[n];
        for(int i = 0; i < n; i++) {
            SummaryStatistics stats = new SummaryStatistics();
            double[] col = X.getColumn(i);
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

        double[][] temp = X.subtract(mu).getData();
        for(int i = 0; i < m; i++)
            for(int j = 0; j < n; j++)
                temp[i][j] /= featureStds[j];

        return MatrixUtils.createRealMatrix(temp);
    }
}
