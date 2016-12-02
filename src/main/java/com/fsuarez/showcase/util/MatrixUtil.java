package com.fsuarez.showcase.util;

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
}
