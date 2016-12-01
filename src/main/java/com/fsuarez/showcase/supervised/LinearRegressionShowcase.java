package com.fsuarez.showcase.supervised;

import java.io.BufferedReader;
import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import com.fsuarez.showcase.Showcase;
import com.fsuarez.showcase.chart.ScatterChart;
import com.fsuarez.showcase.gd.BatchGradientDescent;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.jfree.ui.RefineryUtilities;

/**
 * @author fsuarez
 */
public class LinearRegressionShowcase implements Showcase {

    /**
     *  This kicks off the showcase.
     */
    public void run() {
        double[][] matrixData = readDataFile("lrdata1.txt");
        RealMatrix m = MatrixUtils.createRealMatrix(matrixData);

        ScatterChart chart = new ScatterChart("Population vs Profit");
        chart.createChart(m);
        chart.pack();
        RefineryUtilities.centerFrameOnScreen(chart);
        chart.setVisible(true);

        double[] ones = new  double[m.getRowDimension()];
        Arrays.fill(ones, 1);
        RealMatrix X = MatrixUtils.createRealMatrix(m.getRowDimension(), m.getColumnDimension());
        X.setColumn(0, ones);
        X.setColumnMatrix(1, m.getSubMatrix(0, m.getRowDimension()-1, 0, 0));

        RealVector y = m.getColumnVector(m.getColumnDimension()-1);

        // initialize parameters to 0
        double[] thetaArray = {0.0, 0.0};
        RealMatrix theta = MatrixUtils.createColumnRealMatrix(thetaArray);

        int iterations = 1500;
        double alpha = 0.01;

        BatchGradientDescent gd = new BatchGradientDescent(X, y, theta, iterations, alpha);
        theta = gd.run();

        RealMatrix rawX = m.getSubMatrix(0, m.getRowDimension()-1, 0, 0);
        LinearRegression lr = new UnivariateLinearRegression();
        chart.drawRegressionLine(rawX.getColumnVector(0), lr.predict(X, theta).getColumnVector(0));
    }

    private double[][] readDataFile(String fileName) {

        List<double[]> rows = new ArrayList<>();
        try {
            Path path = Paths.get(getClass().getClassLoader().getResource(fileName).toURI());
            BufferedReader reader = Files.newBufferedReader(path, StandardCharsets.UTF_8);

            String row;
            while ((row = reader.readLine()) != null) {
                String[] columns = row.split(",");
                double[] doubleColumns = new double[columns.length];
                for(int i = 0; i < columns.length; i++)
                    doubleColumns[i] = Double.parseDouble(columns[i]);
                rows.add(doubleColumns);
            }
        } catch (IOException ex) {
            ex.printStackTrace();
        } catch (URISyntaxException e) {
            e.printStackTrace();
        }

        double[][] twoDArray = new double[rows.size()][];
        for(int i = 0; i < rows.size(); i++)
            twoDArray[i] = rows.get(i);

        return twoDArray;
    }
}
