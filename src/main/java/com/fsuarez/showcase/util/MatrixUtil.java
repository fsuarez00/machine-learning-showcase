package com.fsuarez.showcase.util;

import java.util.Arrays;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

/**
 * @author fsuarez
 */
public class MatrixUtil {

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

    public static RealMatrix getThetaZeros(RealMatrix m) {
        double[] thetaArray = new double[m.getColumnDimension()];
        Arrays.fill(thetaArray, 0.0);
        RealMatrix theta = MatrixUtils.createColumnRealMatrix(thetaArray);

        return theta;
    }
}
