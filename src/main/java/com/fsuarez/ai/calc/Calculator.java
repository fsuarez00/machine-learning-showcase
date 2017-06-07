package com.fsuarez.ai.calc;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 * @author fsuarez
 */
public interface Calculator {

    double computeCost(RealMatrix X, RealVector y, RealMatrix theta);

    RealMatrix computePrediction(RealMatrix X, RealMatrix theta);

    RealMatrix computeCostDerivative(RealMatrix X, RealMatrix H, RealVector y);

    RealMatrix computeCostDerivativeRegularization(RealMatrix theta, double lamda, int m);

    double computeCostRegularization(RealMatrix theta, double lambda, int m);

}
