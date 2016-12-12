package com.fsuarez.showcase.chart;

import java.awt.BasicStroke;
import org.apache.commons.math3.linear.RealMatrix;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYItemRenderer;
import org.jfree.chart.renderer.xy.XYSplineRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.ApplicationFrame;

/**
 * @author fsuarez
 */
public class LineChart extends ApplicationFrame {

    public LineChart(String title) {
        super(title);
    }

    public void createChart(RealMatrix m) {
        XYSeries series = new XYSeries("");

        for(int i = 0; i < m.getRowDimension(); i++) {
            double cost = m.getEntry(i, 0);
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
        chartPanel.setPreferredSize(new java.awt.Dimension(500, 270));

        setContentPane(chartPanel);
    }
}
