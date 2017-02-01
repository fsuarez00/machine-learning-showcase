package com.fsuarez.showcase.chart;

import java.awt.Color;
import java.awt.Shape;
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
import org.jfree.util.ShapeUtilities;

/**
 * @author fsuarez
 */
public class ScatterChart extends ApplicationFrame {

    private JFreeChart chart;

    public ScatterChart(String title) {
        super(title);
    }

    public void createChart(RealMatrix m) {
        XYSeries series = new XYSeries("Population vs Profit");

        for(int i = 0; i < m.getRowDimension(); i++) {
            RealVector v = m.getRowVector(i);
            series.add(v.getEntry(0), v.getEntry(1));
        }

        XYSeriesCollection data = new XYSeriesCollection(series);

        chart = ChartFactory.createScatterPlot(
                "",
                "Population of City in 10,000s",
                "Profits in $10,000s",
                data,
                PlotOrientation.VERTICAL,
                false,
                false,
                false
        );

//        DefaultDrawingSupplier.DEFAULT_SHAPE_SEQUENCE[0]
        Shape cross = ShapeUtilities.createDiagonalCross(1, 1);
        XYPlot plot = chart.getXYPlot();
        plot.setDomainCrosshairVisible(true);
        plot.setRangeCrosshairVisible(true);
        XYItemRenderer renderer = plot.getRenderer();
        renderer.setSeriesShape(0, cross);
        renderer.setSeriesPaint(0, Color.red);

        ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new java.awt.Dimension(500, 270));
        setContentPane(chartPanel);
    }

    public void drawRegressionLine(RealVector X, RealVector prediction) {
        XYSeries series = new XYSeries("Population vs Profit: Prediction");

        for(int i = 0; i < X.getDimension(); i++) {
            double x = X.getEntry(i);
            double y = prediction.getEntry(i);
            series.add(x, y);
        }

        XYSeriesCollection data = new XYSeriesCollection(series);

        XYPlot xyPlot = chart.getXYPlot();
        xyPlot.setDataset(1, data);
        XYLineAndShapeRenderer xyLineAndShapeRenderer = new XYLineAndShapeRenderer(true, false);
        xyLineAndShapeRenderer.setSeriesPaint(0, Color.BLUE);
        xyPlot.setRenderer(1, xyLineAndShapeRenderer);
    }
}
