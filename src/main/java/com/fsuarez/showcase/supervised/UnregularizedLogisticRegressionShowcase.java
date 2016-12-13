package com.fsuarez.showcase.supervised;

import com.fsuarez.showcase.Data;
import com.fsuarez.showcase.Learner;
import com.fsuarez.showcase.Showcase;
import org.apache.commons.math3.linear.RealMatrix;

/**
 * @author fsuarez
 */
public class UnregularizedLogisticRegressionShowcase implements Showcase {

    /**
     * @return algorithm data
     */
    @Override
    public Data run() {
        return null;
    }

    @Override
    public void addToChart(RealMatrix X, RealMatrix theta, RealMatrix jHist, Learner learner) {

    }
}
