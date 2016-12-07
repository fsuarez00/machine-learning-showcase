package com.fsuarez.showcase;

import com.fsuarez.showcase.gd.BatchGradientDescent;
import com.fsuarez.showcase.gd.GradientDescent;
import com.fsuarez.showcase.supervised.UnivariateLinearRegression;
import com.fsuarez.showcase.supervised.UnivariateLinearRegressionShowcase;
import org.apache.commons.math3.linear.RealMatrix;

/**
 * @author fsuarez
 */
public class ShowcaseRunner {

    /**
     *  This kicks off the showcase.
     *
     *  @param algorithm
     *  @param gradientDescent
     *  @param iterations used for gradient descent
     *  @param alpha used for gradient descent
     */
    public static void run(String algorithm, String gradientDescent, int iterations, double alpha) {

        Showcase showcase = null;
        Learner learner = null;
        switch(algorithm) {
            case "linear-regression":
                showcase = new UnivariateLinearRegressionShowcase();
                learner = new UnivariateLinearRegression();
        }

        Data data = showcase.run();

        GradientDescent gd = null;
        switch(gradientDescent) {
            case "batch":
                gd = new BatchGradientDescent(data.getX(), data.getY(), data.getTheta(), iterations, alpha, learner);
        }

        RealMatrix learnedTheta = gd.run();

        showcase.addToChart(data.getX(), learnedTheta, learner);
    }

}
