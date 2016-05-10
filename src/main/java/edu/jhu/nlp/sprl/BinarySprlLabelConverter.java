package edu.jhu.nlp.sprl;

/**
 * NA is treated as false as well as all scores that fall below the given
 * threshold; non-na scores at and above the threshold are assigned a positive
 * label
 */
public class BinarySprlLabelConverter implements SprlLabelConverter {
    private double threshold;
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

    @Override
    public double readProb(String label) {
        if (LIKELY.equals(label)) {
            return 5.0;
        } else if (UNLIKELY.equals(label)) {
            return 1.0;
        } else {
            throw new IllegalArgumentException(
                    String.format("provided label doesn't encode an sprl probability acording to %s: %s",
                            BinarySprlLabelConverter.class.toString(), label));
        }
    }

    /**
     * We aren't modeling NA here so everything is applicable
     */
    @Override
    public boolean readApplicable(String label) {
        return true;
    }

}
