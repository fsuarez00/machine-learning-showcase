package com.fsuarez.showcase;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 * @author fsuarez
 */
public class Data {

    private RealMatrix x;
    private RealVector y;
    private RealMatrix theta;

    public Data(RealMatrix x, RealVector y, RealMatrix theta) {
        this.x = x;
        this.y = y;
        this.theta = theta;
    }

    public RealMatrix getX() {
        return x;
    }

    public RealVector getY() {
        return y;
    }

    public RealMatrix getTheta() {
        return theta;
    }
}
