package com.fsuarez.showcase;

import org.apache.commons.math3.linear.RealMatrix;

/**
 * @author fsuarez
 */
public interface Learner {

    RealMatrix predict(RealMatrix X, RealMatrix theta);
}
