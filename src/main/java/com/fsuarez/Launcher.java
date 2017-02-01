package com.fsuarez;

import com.fsuarez.ai.calc.Calculator;
import com.fsuarez.showcase.gd.BatchGradientDescent;
import com.fsuarez.showcase.gd.GradientDescent;
import com.fsuarez.showcase.util.CalculatorFactory;
import com.fsuarez.showcase.util.MatrixUtil;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 * @author fsuarez
 */
public class Launcher {

    public static void main(String... args) throws ParseException {

        Options options = new Options();
        options.addOption("algo", true, "Algorithm to run");
        options.addOption("g", true, "Gradient Descent Algorithm to run");
        options.addOption("i", true, "Number of iterations");
        options.addOption("a", true, "Value of alpha");
        options.addOption("f", true, "File containing the data to be learned");

        CommandLineParser parser = new DefaultParser();
        CommandLine cmd = parser.parse(options, args);

        String fileName = cmd.getOptionValue("f");
        String algorithm = cmd.getOptionValue("algo");
        String gd = cmd.getOptionValue("g");
        String iterString = cmd.getOptionValue("i");
        String alphaString = cmd.getOptionValue("a");
        int iterations = 0;
        if(iterString != null)
            iterations = Integer.parseInt(cmd.getOptionValue("i"));
        double alpha = 0.0;
        if(alphaString != null)
            alpha = Double.parseDouble(cmd.getOptionValue("a"));

        Calculator calculator = CalculatorFactory.getCalculator(algorithm);
        RealMatrix mData = MatrixUtil.readDataFile(fileName);

        RealMatrix X = MatrixUtil.appendBiasTermColumnWithOnes(mData.getSubMatrix(0, mData.getRowDimension()-1, 0, 0));
        RealVector y = mData.getColumnVector(mData.getColumnDimension()-1);
        RealMatrix theta = MatrixUtil.getThetaZeros(X);

        GradientDescent gradientDescent = null;
        switch(gd) {
            case "batch":
                gradientDescent = new BatchGradientDescent(X, y, theta, iterations, alpha, calculator);
        }
        RealMatrix learnedTheta = gradientDescent.run();

        // TODO get user input and output predicted data
    }

}
