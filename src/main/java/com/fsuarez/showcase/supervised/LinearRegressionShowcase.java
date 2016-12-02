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
import com.fsuarez.showcase.util.MatrixUtil;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.jfree.ui.RefineryUtilities;

import static com.sun.tools.doclint.Entity.mu;

/**
 * @author fsuarez
 */
public class LinearRegressionShowcase implements Showcase {

    /**
     *  This kicks off the showcase.
     */
    public void run() {
        // Univariate Linear Regression
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

        BatchGradientDescent gd = new BatchGradientDescent(X, y, theta, iterations, alpha, new UnivariateLinearRegression());
        theta = gd.run();

        RealMatrix rawX = m.getSubMatrix(0, m.getRowDimension()-1, 0, 0);
        LinearRegression lr = new UnivariateLinearRegression();
        chart.drawRegressionLine(rawX.getColumnVector(0), lr.predict(X, theta).getColumnVector(0));

        // Multivariate Linear Regression
        matrixData = readDataFile("lrdata2.txt");
        m = MatrixUtils.createRealMatrix(matrixData);
        X = MatrixUtils.createRealMatrix(m.getRowDimension(), m.getColumnDimension()-1);
        X.setColumnMatrix(0, m.getSubMatrix(0, m.getRowDimension()-1, 0, 0));
        X.setColumnMatrix(1, m.getSubMatrix(0, m.getRowDimension()-1, 1, 1));

        y = m.getColumnVector(m.getColumnDimension()-1);

        System.out.println("Feature Normalization...");
        RealMatrix xNorm = featureNormmalize(X);

        X = MatrixUtils.createRealMatrix(m.getRowDimension(), m.getColumnDimension());
        ones = new  double[m.getRowDimension()];
        Arrays.fill(ones, 1);
        X.setColumn(0, ones);
        for(int i = 0; i < m.getColumnDimension()-1; i++)
            X.setColumnMatrix(i+1, xNorm.getSubMatrix(0, xNorm.getRowDimension()-1, i, i));

        System.out.println(MatrixUtil.toString(X));
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

    private RealMatrix featureNormmalize(RealMatrix X) {
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
