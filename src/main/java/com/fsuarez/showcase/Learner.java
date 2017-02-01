package com.fsuarez.showcase;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 * @author fsuarez
 */
public interface Learner {

    double computeCost(RealMatrix X, RealVector y, RealMatrix theta);

    RealMatrix predict(RealMatrix X, RealMatrix theta);
}
