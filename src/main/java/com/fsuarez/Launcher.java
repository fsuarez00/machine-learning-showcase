package com.fsuarez;

import com.fsuarez.showcase.ShowcaseRunner;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

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

        CommandLineParser parser = new DefaultParser();
        CommandLine cmd = parser.parse(options, args);

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

        ShowcaseRunner.run(algorithm, gd, iterations, alpha);
    }

}
