package com.fsuarez.showcase.supervised;

import java.util.Arrays;
import com.fsuarez.showcase.Data;
import com.fsuarez.showcase.Learner;
import com.fsuarez.showcase.Showcase;
import com.fsuarez.showcase.chart.ScatterChart;
import com.fsuarez.showcase.util.MatrixUtil;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.jfree.ui.RefineryUtilities;

/**
 * @author fsuarez
 */
public class UnivariateLinearRegressionShowcase implements Showcase {

    private ScatterChart chart;

    /**
     *  @return algorithm data
     */
    public Data run() {
        // Univariate Linear Regression
        double[][] matrixData = readDataFile("lrdata1.txt");
        RealMatrix m = MatrixUtils.createRealMatrix(matrixData);

        chart = new ScatterChart("Population vs Profit");
        chart.createChart(m);
        chart.pack();
        RefineryUtilities.centerFrameOnScreen(chart);
        chart.setVisible(true);

        RealMatrix X = MatrixUtil.appendBiasTermColumnWithOnes(m.getSubMatrix(0, m.getRowDimension()-1, 0, 0));

        RealVector y = m.getColumnVector(m.getColumnDimension()-1);

        // initialize parameters to 0
        RealMatrix theta = MatrixUtil.getThetaZeros(X);

        return new Data(X, y, theta);
    }

    @Override
    public void addToChart(RealMatrix X, RealMatrix theta, RealMatrix jHist, Learner learner) {
        // removes column with 1's
        RealMatrix rawX = X.getSubMatrix(0, X.getRowDimension() - 1, 1, 1);
        chart.drawRegressionLine(rawX.getColumnVector(0), learner.predict(X, theta).getColumnVector(0));
    }
}
