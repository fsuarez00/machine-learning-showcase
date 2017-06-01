package com.fsuarez.showcase;

import java.util.Arrays;
import com.fsuarez.ai.calc.LinearRegressionCalculator;
import com.fsuarez.showcase.gd.BatchGradientDescent;
import com.fsuarez.showcase.gd.GradientDescent;
import com.fsuarez.showcase.util.MatrixUtil;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYSplineRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RefineryUtilities;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author fsuarez
 */
public class MultivariateLinearRegressionShowcase {

    private static final Logger LOGGER = LoggerFactory.getLogger(MultivariateLinearRegressionShowcase.class);

    public static void main(String... args) {
        // initialize parameters
        RealMatrix mData = MatrixUtil.readDataFile("src/test/resources/lrdata2.txt");

        RealMatrix rawX = mData.getSubMatrix(0, mData.getRowDimension()-1, 0, mData.getColumnDimension()-2);

        LOGGER.info("Feature Normalization...");
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
        RealVector yVector = mData.getColumnVector(mData.getColumnDimension()-1);
        RealMatrix theta = MatrixUtil.getThetaZeros(xMatrix);

        // manually run gradient descent to capture cost per iteration
        int iterations = 1500;
        LinearRegressionCalculator linearRegressionCalculator = new LinearRegressionCalculator();
        GradientDescent gradientDescent = new BatchGradientDescent(xMatrix, yVector, theta, iterations, 0.01, linearRegressionCalculator);
        RealMatrix J = MatrixUtils.createRealMatrix(iterations, 1);
        for(int i = 0; i < iterations; i++) {
            theta = gradientDescent.descend(theta);
            J.setEntry(i, 0, linearRegressionCalculator.computeCost(xMatrix, yVector, theta));
        }

        LOGGER.info("Theta found by gradient descent:\n\n{}", MatrixUtil.toString(theta));

        // generate chart
        ApplicationFrame applicationFrame = new ApplicationFrame("Learning Curve");
        XYSeries series = new XYSeries("");
        for(int i = 0; i < J.getRowDimension(); i++) {
            double cost = J.getEntry(i, 0);
            series.add(i, cost);
        }
        XYSeriesCollection data = new XYSeriesCollection(series);
        JFreeChart chart = ChartFactory.createXYLineChart(
                "",
                "Number of iterations",
                "Cost J",
                data,
                PlotOrientation.VERTICAL,
                false,
                false,
                false
        );
        XYPlot plot = chart.getXYPlot();
        XYSplineRenderer renderer = new XYSplineRenderer();
        renderer.setSeriesShapesVisible(0, false);
        plot.setRenderer(renderer);
        ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new java.awt.Dimension(1000, 540));
        applicationFrame.setContentPane(chartPanel);
        applicationFrame.pack();
        RefineryUtilities.centerFrameOnScreen(applicationFrame);
        applicationFrame.setVisible(true);
    }
}
