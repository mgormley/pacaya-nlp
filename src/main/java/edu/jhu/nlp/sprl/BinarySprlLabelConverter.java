package edu.jhu.nlp.sprl;

import java.util.Arrays;
import java.util.Set;
import java.util.TreeSet;

/**
 * NA is treated as false as well as all scores that fall below the given
 * threshold; non-na scores at and above the threshold are assigned a positive
 * label
 */
public class BinarySprlLabelConverter implements SprlLabelConverter {
    private double threshold;
    private static Set<String> validLabels = new TreeSet<>(Arrays.asList(LIKELY, UNLIKELY));

    public BinarySprlLabelConverter(double threshold) {
        this.threshold = threshold;
    }

    @Override
    public String toLabel(double probability, boolean applicable) {
        if (applicable && probability >= threshold) {
            return LIKELY;
        } else {
            return UNLIKELY;
        }
    }

    private void assertValid(String label) {
        if (!validLabels.contains(label)) {
            throw new IllegalArgumentException(String.format("provided label isn't valid for converter %s: %s",
                    BinarySprlLabelConverter.class.toString(), label));
        }
    }

    @Override
    public double readProb(String label) {
        assertValid(label);
        if (LIKELY.equals(label)) {
            return 5.0;
        } else {
            return 1.0;
        }
    }

    /**
     * We aren't modeling NA here so everything is applicable
     */
    @Override
    public boolean readApplicable(String label) {
        assertValid(label);
        return true;
    }

}
