package com.fsuarez.showcase.supervised;

import com.fsuarez.showcase.Learner;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 * @author fsuarez
 */
public interface LogisticRegression extends Learner {

    RealMatrix computeGradient(RealMatrix X, RealMatrix H, RealVector y);
}
