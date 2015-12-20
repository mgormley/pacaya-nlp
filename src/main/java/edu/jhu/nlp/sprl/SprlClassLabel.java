package edu.jhu.nlp.sprl;

import java.util.ArrayList;

public enum SprlClassLabel {
    UNLIKELY, UNKNOWN, LIKELY, NOT_AN_ARG;

    // add the labels as string names
    public static ArrayList<String> sprlLabels;
    static {
        sprlLabels = new ArrayList<>();
        for (SprlClassLabel label : values()) {
            sprlLabels.add(label.name());
        }
    }

    public static SprlClassLabel getLabel(Double p) {
        // if (p == null)
        // return SPRLClassLabel.UNKNOWN;
        if (p < 3) {
            return UNLIKELY;
        } else if (p > 3) {
            return LIKELY;
        } else {
            return UNKNOWN;
        }
    }

    public static double getResponse(SprlClassLabel label) {
        return getResponse(label.ordinal());
    }

    public static double getResponse(int label) {
        if (label == UNLIKELY.ordinal()) {
            return 1;
        } else if (label == LIKELY.ordinal()) {
            return 5;
        } else if (label == UNKNOWN.ordinal()) {
            return 3;
        } else {
            return 0;
        }
    }


}