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
import java.util.LinkedList;
import java.util.List;
import com.fsuarez.showcase.Showcase;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

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
