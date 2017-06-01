package com.fsuarez.showcase.util;

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
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

/**
 * @author fsuarez
 */
public class MatrixUtil {

    private MatrixUtil(){}

    public static String toString(RealMatrix m) {
        StringBuilder stringBuilder = new StringBuilder();
        double[][] data = m.getData();

        for(int i = 0; i < m.getRowDimension(); i++) {
            stringBuilder.append("| ");
            for(int j = 0; j < m.getColumnDimension(); j++)
                stringBuilder.append(data[i][j]).append(' ');
            stringBuilder.append("|\n");
        }

        return stringBuilder.toString();
    }

    public static RealMatrix appendBiasTermColumnWithOnes(RealMatrix m) {
        double[] ones = new  double[m.getRowDimension()];
        Arrays.fill(ones, 1.0);

        RealMatrix newM = MatrixUtils.createRealMatrix(m.getRowDimension(), m.getColumnDimension() + 1);
        newM.setColumn(0, ones);
        for(int i = 0; i < m.getColumnDimension(); i++)
            newM.setColumnMatrix(i+1, m.getSubMatrix(0, m.getRowDimension()-1, i, i));

        return newM;
    }

    public static RealMatrix getThetaZeros(int dimensions) {
        double[] thetaArray = new double[dimensions];
        Arrays.fill(thetaArray, 0.0);
        RealMatrix theta = MatrixUtils.createColumnRealMatrix(thetaArray);

        return theta;
    }

    public static RealMatrix readDataFile(String fileName) {

        List<double[]> rows = new ArrayList<>();
        try {
            Path path = Paths.get(MatrixUtil.class.getClassLoader().getResource(fileName).toURI());
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

        return MatrixUtils.createRealMatrix(twoDArray);
    }
}
