package com.fsuarez.showcase.util;

import com.fsuarez.ai.calc.Calculator;
import com.fsuarez.ai.calc.LinearRegressionCalculator;

/**
 * @author fsuarez
 */
public class CalculatorFactory {

    private CalculatorFactory(){}

    public static Calculator getCalculator(String algorithm) {
        switch(algorithm) {
            case "linear-regression":
                return new LinearRegressionCalculator();
            case "multi-linear-regression":
                return new LinearRegressionCalculator();
            default:
                return null;
        }
    }
}
