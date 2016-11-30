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
        options.addOption("a", true, "Algorithm to run");

        CommandLineParser parser = new DefaultParser();
        CommandLine cmd = parser.parse(options, args);

        String algorithm = cmd.getOptionValue("a");

        ShowcaseRunner.run(algorithm);
    }

}
