package com.fsuarez.showcase.util;

import com.fsuarez.ai.calc.Calculator;
import com.fsuarez.ai.calc.LinearRegressionCalculator;
import com.fsuarez.ai.calc.LogisticRegressionCalculator;
import com.fsuarez.ai.calc.NormalEquationLinearRegressionCalculator;

/**
 * @author fsuarez
 */
public class CalculatorFactory {

    public static final String LINEAR_REGRESSION = "linear-regression";
    public static final String LINEAR_REGRESSION_NORMAL_EQ = "linear-regression-norm-eq";
    public static final String LOGISTIC_REGRESSION = "logistic-regression";

    private CalculatorFactory(){}

    public static Calculator getCalculator(String algorithm) {
        switch(algorithm) {
            case LINEAR_REGRESSION:
                return new LinearRegressionCalculator();
            case LOGISTIC_REGRESSION:
                return new LogisticRegressionCalculator();
            default:
                return null;
        }
    }
}
