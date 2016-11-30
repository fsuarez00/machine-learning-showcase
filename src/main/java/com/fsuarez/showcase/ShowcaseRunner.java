package com.fsuarez.showcase;

import com.fsuarez.showcase.supervised.LinearRegressionShowcase;

/**
 * @author fsuarez
 */
public class ShowcaseRunner {

    /**
     *  This kicks off the showcase.
     */
    public static void run(String algorithm) {
        switch(algorithm) {
            case "linear-regression":
                new LinearRegressionShowcase().run();
        }
    }

}
