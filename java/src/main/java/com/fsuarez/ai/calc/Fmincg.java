package com.fsuarez.ai.calc;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author fsuarez
 */
public class Fmincg {

    private static final Logger LOGGER = LoggerFactory.getLogger(Fmincg.class);

    private Fmincg(){};

    public static FmincgReturn fMinUnc(Calculator calculator, RealMatrix X, RealVector y, RealMatrix theta, double lambda, int iterations) {
        int length = iterations;
        int m = X.getRowDimension();

        RealMatrix fTheta = null;

        //a bunch of constants for line searches
        double RHO = 0.01;  //RHO and SIG are the constants in the Wolfe-Powell conditions
        double SIG = 0.5;
        double INT = 0.1;   //don't reevaluate within 0.1 of the limit of the current bracket
        double EXT = 3.0;   //extrapolate maximum 3 times the current bracket
        double MAX = 20.0;    //max 20 function evaluations per line search
        double RATIO = 100.0; //maximum allowed slope ratio

        double red = 1.0;

        int i = 0;  //zero the run length counter
        boolean lsFailed = false;   //no previous line search has failed
        double f1 = calculator.computeCost(X, y, theta) + calculator.computeCostRegularization(theta, lambda, m);    //get function value and gradient
        RealMatrix df1 = calculator.computeCostDerivative(X, calculator.computePrediction(X, theta), y).add(calculator.computeCostDerivativeRegularization(theta, lambda,m));
        RealMatrix s = df1.scalarMultiply(-1.0); //search direction is steepest
        double d1 = s.transpose().multiply(s).scalarMultiply(-1.0).getData()[0][0];
        double z1 = red/(1.0-d1); //initial step is red/(|s|+1)

        while(i < length) {
            i++;

            // copy current values
            RealMatrix X0 = theta.copy();
            double f0 = f1;
            RealMatrix df0 = df1.copy();
            // begin line search
            theta = theta.add(s.scalarMultiply(z1));

            double f2 = calculator.computeCost(X, y, theta) + calculator.computeCostRegularization(theta, lambda, m);
            RealMatrix df2 = calculator.computeCostDerivative(X, calculator.computePrediction(X, theta), y).add(calculator.computeCostDerivativeRegularization(theta, lambda,m));
            double d2 = df2.transpose().multiply(s).getData()[0][0];
            //initialize point 3 equal to point 1
            double f3 = f1;
            double d3 = d1;
            double z3 = -z1;
            double M = MAX;
            boolean success = false;
            double limit = -1;
            //initialize quantities
            while(true) {
                while (((f2 > (f1 + z1 * RHO * d1)) || (d2 > (-SIG * d1))) && (M > 0.0)) {
                    limit = z1; //tighten the bracket
                    double z2;
                    if (f2 > f1)
                        z2 = z3 - (0.5 * d3 * z3 * z3) / (d3 * z3 + f2 - f3); //quadratic fit
                    else {
                        double A = 6.0 * (f2 - f3) / z3 + 3.0 * (d2 + d3);  //cubic fit
                        double B = 3.0 * (f3 - f2) - z3 * (d3 + 2.0 * d2);
                        z2 = (Math.sqrt(B * B - A * d2 * z3 * z3) - B) / A;    //numerical error possible
                    }
                    if (Double.isNaN(z2) || Double.isInfinite(z2))
                        z2 = z3 / 2;  // if we had a numerical problem then bisect
                    z2 = Math.max(Math.min(z2, INT * z3), (1 - INT) * z3); // don't accept too close to limits
                    z1 = z1 + z2;   // update the step
                    theta = theta.add(s.scalarMultiply(z2));
                    f2 = calculator.computeCost(X, y, theta) + calculator.computeCostRegularization(theta, lambda, m);
                    df2 = calculator.computeCostDerivative(X, calculator.computePrediction(X, theta), y).add(calculator.computeCostDerivativeRegularization(theta, lambda,m));

                    d2 = df2.transpose().multiply(s).getData()[0][0];
                    z3 = z3 - z2; // z3 is now relative to the location of z2

                    M--;
                }
                if (f2 > f1 + z1 * RHO * d1 || d2 > -SIG * d1)
                    break;  // this is a failure
                if (d2 > SIG * d1) {
                    success = true;
                    break;  // % success
                }
                if (M == 0.0)
                    break;  // failure

                double A = 6.0 * (f2 - f3) / z3 + 3.0 * (d2 + d3);  // make cubic extrapolation
                double B = 3.0 * (f3 - f2) - z3 * (d3 + 2.0 * d2);
                double z2 = -d2 * z3 * z3 / (B + Math.sqrt(B * B - A * d2 * z3 * z3));    // num. error possible
                if (Double.isNaN(z2) || Double.isInfinite(z2) || z2 < 0) {    // % num prob or wrong sign?
                    if (limit < -0.5)   // if we have no upper limit
                        z2 = z1 * (EXT - 1.0);  // the extrapolate the maximum amount
                    else
                        z2 = (limit - z1) / 2.0;  // otherwise bisect
                } else if ((limit > -0.5) && (z2 + z1 > limit))  // extrapolation beyond max?
                    z2 = (limit - z1) / 2;  //bisect
                else if ((limit < -0.5) && (z2 + z1 > z1 * EXT)) // extrapolation beyond limit
                    z2 = z1 * (EXT - 1.0);  // set to extrapolation limit
                else if (z2 < -z3 * INT)
                    z2 = -z3 * INT;
                else if ((limit > -0.5) && (z2 < (limit - z1) * (1.0 - INT)))  // too close to limit?
                    z2 = (limit - z1) * (1.0 - INT);

                // set point 3 equal to point 2
                f3 = f2;
                d3 = d2;
                z3 = -z2;
                z1 = z1 + z2;

                // update current estimates
                theta = theta.add(s.scalarMultiply(z2));
                f2 = calculator.computeCost(X, y, theta) + calculator.computeCostRegularization(theta, lambda, m);
                df2 = calculator.computeCostDerivative(X, calculator.computePrediction(X, theta), y).add(calculator.computeCostDerivativeRegularization(theta, lambda,m));
                d2 = df2.transpose().multiply(s).getData()[0][0];

                M--;
            }   // end of line search

            //if line search succeeded
            if(success) {
                f1 = f2;
                if(fTheta == null)
                    fTheta = MatrixUtils.createRealMatrix(new double[][]{{f1}});
                else {
                    double[][] data = fTheta.getData();
                    double[][] newData = new double[data.length][data[0].length + 1];
                    for(int o=0; o < data.length; o++) {
                        for(int n = 0; n < data[0].length; n++)
                            newData[o][n] = data[o][n];
                        newData[o][data.length] = f1;
                    }
                    fTheta = MatrixUtils.createRealMatrix(newData);
                }
                LOGGER.info("Iteration: {} | Cost: {}", i, f1);
                double part1 = df2.transpose().multiply(df2).subtract(df1.transpose().multiply(df2)).getData()[0][0];
                double part2 = df1.transpose().multiply(df1).getData()[0][0];
                s = s.scalarMultiply(part1/part2).subtract(df2);    // Polak-Ribiere direction
                // swap derivatives
                RealMatrix tmp = df1;
                df1 = df2;
                df2 = tmp;
                d2 = df1.transpose().multiply(s).getData()[0][0];

                // new slope must be negative
                if(d2 > 0.0){
                    s = df1.scalarMultiply(-1.0);   // otherwise use steepest direction
                    d2 = s.transpose().multiply(s).scalarMultiply(-1.0).getData()[0][0];
                }

                // slope ratio but max RATIO, 0 supposed to be realmin(2.2251e-308 for double precision and 1.1755e-38 for single precision)
                z1 = z1 * Math.min(RATIO, d1/(d2-0.0));
                d1 = d2;
                // this line search did not fail
                lsFailed = false;
            } else {
                //restore point from before failed line search
                theta = X0;
                f1 = f0;
                df1 = df0;
                // line search failed twice in a row or we ran out of time, so we give up
                if(lsFailed || i > length)
                    break;

                // swap derivatives
                RealMatrix tmp = df1;
                df1 = df2;
                df2 = tmp;
                s = df1.scalarMultiply(-1.0);   // try steepest
                d1 = s.transpose().multiply(s).scalarMultiply(-1.0).getData()[0][0];
                z1 = 1.0/(1.0-d1);
                //this line search failed
                lsFailed = true;
            }
        }

        return new FmincgReturn(theta, fTheta, i);
    }

    public static class FmincgReturn {

        private RealMatrix theta;

        private RealMatrix fTheta;

        private int i;

        public FmincgReturn(RealMatrix theta, RealMatrix fTheta, int i){
            this.theta = theta;
            this.fTheta = fTheta;
            this.i = i;
        }

        public RealMatrix getTheta(){
            return this.theta;
        }

        public RealMatrix getfTheta(){
            return this.fTheta;
        }

        public int getI(){
            return this.i;
        }

    }
}
