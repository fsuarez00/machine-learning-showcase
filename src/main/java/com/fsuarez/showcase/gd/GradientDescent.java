package com.fsuarez.showcase.gd;

import java.util.Map;
import org.apache.commons.math3.linear.RealMatrix;

/**
 * @author fsuarez
 */
public interface GradientDescent {

    Map<String, RealMatrix> run();
}
