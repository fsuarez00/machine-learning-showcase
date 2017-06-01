package com.fsuarez.showcase;

import java.awt.Color;
import java.awt.Shape;
import com.fsuarez.ai.calc.LinearRegressionCalculator;
import com.fsuarez.showcase.gd.BatchGradientDescent;
import com.fsuarez.showcase.gd.GradientDescent;
import com.fsuarez.showcase.util.MatrixUtil;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYItemRenderer;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RefineryUtilities;
import org.jfree.util.ShapeUtilities;

/**
 * @author fsuarez
 */
public class UnivariateLinearRegressionShowcase {

    public static void main(String... args) {
        RealMatrix mData = MatrixUtil.readDataFile("src/test/resources/lrdata1.txt");

        // generate chart
        ApplicationFrame applicationFrame = new ApplicationFrame("Population vs Profit");
        XYSeries series1 = new XYSeries("Population vs Profit");
        for(int i = 0; i < mData.getRowDimension(); i++) {
            RealVector v = mData.getRowVector(i);
            series1.add(v.getEntry(0), v.getEntry(1));
        }
        XYSeriesCollection data1 = new XYSeriesCollection(series1);
        JFreeChart chart = ChartFactory.createScatterPlot(
                "",
                "Population of City in 10,000s",
                "Profits in $10,000s",
                data1,
                PlotOrientation.VERTICAL,
                false,
                false,
                false
        );
        Shape cross = ShapeUtilities.createDiagonalCross(1, 1);
        XYPlot plot = chart.getXYPlot();
        plot.setDomainCrosshairVisible(true);
        plot.setRangeCrosshairVisible(true);
        XYItemRenderer renderer = plot.getRenderer();
        renderer.setSeriesShape(0, cross);
        renderer.setSeriesPaint(0, Color.red);

        ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new java.awt.Dimension(1000, 540));
        applicationFrame.setContentPane(chartPanel);
        applicationFrame.pack();
        RefineryUtilities.centerFrameOnScreen(applicationFrame);
        applicationFrame.setVisible(true);

        // initialize parameters
        RealMatrix rawX = mData.getSubMatrix(0, mData.getRowDimension()-1, 0, mData.getColumnDimension()-2);
        RealMatrix xMatrix = MatrixUtil.appendBiasTermColumnWithOnes(rawX);
        RealVector yVector = mData.getColumnVector(mData.getColumnDimension()-1);
        RealMatrix theta = MatrixUtil.getThetaZeros(xMatrix);

        // run gradient descent
        LinearRegressionCalculator linearRegressionCalculator = new LinearRegressionCalculator();
        GradientDescent gradientDescent = new BatchGradientDescent(xMatrix, yVector, theta, 1500, 0.01, linearRegressionCalculator);
        RealMatrix learnedTheta = gradientDescent.run();

        // draw regression line
        XYSeries series2 = new XYSeries("Population vs Profit: Prediction");
        RealVector rawXColumnVector = rawX.getColumnVector(0);
        RealVector predictionColumnVector = linearRegressionCalculator.computePrediction(xMatrix, learnedTheta).getColumnVector(0);
        for(int i = 0; i < rawXColumnVector.getDimension(); i++) {
            double x = rawXColumnVector.getEntry(i);
            double y = predictionColumnVector.getEntry(i);
            series2.add(x, y);
        }
        XYSeriesCollection data2 = new XYSeriesCollection(series2);
        XYPlot xyPlot = chart.getXYPlot();
        xyPlot.setDataset(1, data2);
        XYLineAndShapeRenderer xyLineAndShapeRenderer = new XYLineAndShapeRenderer(true, false);
        xyLineAndShapeRenderer.setSeriesPaint(0, Color.BLUE);
        xyPlot.setRenderer(1, xyLineAndShapeRenderer);
    }

}
