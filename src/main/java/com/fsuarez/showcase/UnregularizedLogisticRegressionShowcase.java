package com.fsuarez.showcase;

import java.awt.Color;
import java.awt.Shape;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import com.fsuarez.ai.calc.LogisticRegressionCalculator;
import com.fsuarez.showcase.gd.BatchGradientDescent;
import com.fsuarez.showcase.gd.GradientDescent;
import com.fsuarez.showcase.util.MatrixUtil;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.linear.MatrixUtils;
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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author fsuarez
 */
public class UnregularizedLogisticRegressionShowcase {

    private static final Logger LOGGER = LoggerFactory.getLogger(UnregularizedLogisticRegressionShowcase.class);

    public static void main(String... args) {
        // initialize parameters
        RealMatrix mData = MatrixUtil.readDataFile("src/test/resources/logrdata1.txt");
        RealMatrix rawX = mData.getSubMatrix(0, mData.getRowDimension()-1, 0, mData.getColumnDimension()-2);
        RealMatrix xMatrix = MatrixUtil.appendBiasTermColumnWithOnes(rawX);
        RealVector yVector = mData.getColumnVector(mData.getColumnDimension()-1);
        RealMatrix theta = MatrixUtil.getThetaZeros(xMatrix);

        // generate chart
        ApplicationFrame applicationFrame = new ApplicationFrame("Exam 1 vs Exam 2");
        XYSeries series1 = new XYSeries("Admitted");
        for(int i = 0; i < rawX.getRowDimension(); i++) {
            RealVector v = rawX.getRowVector(i);
            if(yVector.getEntry(i) == 1.0)
                series1.add(v.getEntry(0), v.getEntry(1));
        }
        XYSeries series2 = new XYSeries("Not Admitted");
        for(int i = 0; i < rawX.getRowDimension(); i++) {
            RealVector v = rawX.getRowVector(i);
            if(yVector.getEntry(i) == 0.0)
                series2.add(v.getEntry(0), v.getEntry(1));
        }
        XYSeriesCollection data = new XYSeriesCollection(series1);
        data.addSeries(series2);
        JFreeChart chart = ChartFactory.createScatterPlot(
                "",
                "Exam 1 Score",
                "Exam 2 Score",
                data,
                PlotOrientation.VERTICAL,
                true,
                false,
                false
        );
        Shape cross = ShapeUtilities.createDiagonalCross(1, 1);
        XYPlot plot = chart.getXYPlot();
        plot.setDomainCrosshairVisible(true);
        plot.setRangeCrosshairVisible(true);
        XYItemRenderer renderer = plot.getRenderer();
        renderer.setSeriesShape(0, cross);
        renderer.setSeriesPaint(0, Color.BLACK);
        renderer.setSeriesShape(1, cross);
        renderer.setSeriesPaint(1, Color.yellow);

        ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new java.awt.Dimension(1000, 540));
        applicationFrame.setContentPane(chartPanel);
        applicationFrame.pack();
        RefineryUtilities.centerFrameOnScreen(applicationFrame);
        applicationFrame.setVisible(true);

        // run gradient descent
        LogisticRegressionCalculator logisticRegressionCalculator = new LogisticRegressionCalculator();
        LOGGER.info("Cost before training: {}", logisticRegressionCalculator.computeCost(xMatrix, yVector, theta));
        GradientDescent gradientDescent = new BatchGradientDescent(xMatrix, yVector, theta, 20, 0.001, logisticRegressionCalculator);
        RealMatrix learnedTheta = gradientDescent.run();

        LOGGER.info("Cost after training: {}", logisticRegressionCalculator.computeCost(xMatrix, yVector, learnedTheta));

        RealMatrix inputX = MatrixUtil.appendBiasTermColumnWithOnes(MatrixUtils.createRowRealMatrix(new double[]{45.0, 85.0}));
        RealMatrix predictionMatrix = logisticRegressionCalculator.computePrediction(inputX, learnedTheta);
        RealMatrix P = MatrixUtils.createRealMatrix(inputX.getRowDimension(), 1);
        for(int i = 0; i < inputX.getRowDimension(); i++)
            if(predictionMatrix.getEntry(i, 0) >= 0.5)
                P.setEntry(i, 0, 1.0);
            else
                P.setEntry(i, 0, 0.0);
        LOGGER.info("Prediction for {}: {}", MatrixUtil.toString(inputX), P.getEntry(0, 0));
        // plot decision boundary -- broken at the moment
        // plotDecisionBoundary(rawX, learnedTheta, plot);
    }

    private static void plotDecisionBoundary(RealMatrix rawX, RealMatrix learnedTheta, XYPlot plot) {
        //only need 2 points so get the lowest and highest x
        List<Double> xColumnList = Arrays.asList(ArrayUtils.toObject(rawX.getColumnVector(0).toArray()));
        int minIndex = xColumnList.indexOf(Collections.min(xColumnList));
        int maxIndex = xColumnList.indexOf(Collections.max(xColumnList));
        double minX = xColumnList.get(minIndex)-2.0;
        double maxX = xColumnList.get(maxIndex)+2.0;
        // get y
        double yIntercept = -learnedTheta.getEntry(0, 0) / learnedTheta.getEntry(2, 0);
        double slope = -learnedTheta.getEntry(1, 0) / learnedTheta.getEntry(2, 0);
        double y1 = slope * minX + yIntercept;
        double y2 = slope * maxX + yIntercept;

        LOGGER.info("Slope: [{}] Intercept: [{}]", slope, yIntercept);

        XYSeries series3 = new XYSeries("Decision Boundary");
        series3.add(minX, y1);
        series3.add(maxX, y2);
        XYSeriesCollection data2 = new XYSeriesCollection(series3);
        plot.setDataset(1, data2);
        XYLineAndShapeRenderer xyLineAndShapeRenderer = new XYLineAndShapeRenderer(true, false);
        xyLineAndShapeRenderer.setSeriesPaint(0, Color.RED);
        plot.setRenderer(1, xyLineAndShapeRenderer);
    }
}
