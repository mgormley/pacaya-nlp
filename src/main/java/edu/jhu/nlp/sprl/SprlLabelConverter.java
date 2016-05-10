package edu.jhu.nlp.sprl;

public interface SprlLabelConverter {

    public String toLabel(double probability, boolean applicable);
    public double readProb(String label);
    public boolean readApplicable(String label);
    public final static String NA = "NA";
    public final static String LIKELY = "LIKELY";
    public final static String UNLIKELY = "UNLIKELY";
    public final static String NOT_AN_ARG = "NOT_AN_ARG";
    public final static String UNKNOWN = "UNKNOWN";
    
    public static String nil() { 
        return NOT_AN_ARG;
    }

    public default boolean isNil(String label) { 
        return nil().equals(label);
    }

    public default String toLabel(double probability, double applicable) {
        if (applicable == 0.0) {
            throw new IllegalArgumentException("applicable signum need to be positive or negative (not 0.0)");
        }
        return toLabel(probability, applicable > 0.0);
    }

}
