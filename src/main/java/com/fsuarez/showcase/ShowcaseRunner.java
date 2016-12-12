package com.fsuarez.showcase;

import java.util.Map;
import com.fsuarez.showcase.gd.BatchGradientDescent;
import com.fsuarez.showcase.gd.GradientDescent;
import com.fsuarez.showcase.supervised.MultivariateLinearRegression;
import com.fsuarez.showcase.supervised.MultivariateLinearRegressionShowcase;
import com.fsuarez.showcase.supervised.NormalEquationLinearRegression;
import com.fsuarez.showcase.supervised.NormalEquationLinearRegressionShowcase;
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
                break;
            case "multi-linear-regression":
                showcase = new MultivariateLinearRegressionShowcase();
                learner = new MultivariateLinearRegression();
                break;
            case "linear-regression-norm-eq":
                showcase = new NormalEquationLinearRegressionShowcase();
                // no need to run gradient descent due to closed-form solution
                Data data = showcase.run();
                showcase.addToChart(data.getX(), data.getTheta(), null, learner);
                return;
            default:
                return;
        }

        Data data = showcase.run();

        GradientDescent gd = null;
        switch(gradientDescent) {
            case "batch":
                gd = new BatchGradientDescent(data.getX(), data.getY(), data.getTheta(), iterations, alpha, learner);
        }

        Map<String, RealMatrix> resultMap = gd.run();

        RealMatrix learnedTheta = resultMap.get("theta");
        RealMatrix jHist = resultMap.get("jHist");

        showcase.addToChart(data.getX(), learnedTheta, jHist, learner);
    }

}
