package edu.jhu.nlp.sprl;

import java.util.ArrayList;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.pacaya.util.cli.Opt;

public enum SprlClassLabel {
    UNLIKELY, LIKELY, NOT_AN_ARG, UNKNOWN, NA;

    public static enum SplitMode {
        Split_12_3_45, Split_123_45, Split_1_234_5, Split_1234_5,
    }

    private static final Logger log = LoggerFactory.getLogger(SprlClassLabel.class);

    @Opt(hasArg = true, description = "how to convert likert scale responses to class labels")
    public static SplitMode splitMode = SplitMode.Split_12_3_45;

    @Opt(hasArg = true, description = "whether to treat non-positive scores as NA")
    public static boolean modelNA = false;
    
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
        if (p < 0 && modelNA) {
            return SprlClassLabel.NA;
        }
        switch (splitMode) {
        case Split_123_45:
            if (p < 4) {
                return UNLIKELY;
            } else {
                return LIKELY;
            }
        case Split_1_234_5:
            if (p < 2) {
                return UNLIKELY;
            } else if (p > 4) {
                return LIKELY;
            } else {
                return UNKNOWN;
            }
        case Split_1234_5:
            if (p < 5) {
                return UNLIKELY;
            } else {
                return LIKELY;
            }
        default:
            assert splitMode == SplitMode.Split_12_3_45;
            if (p < 3) {
                return UNLIKELY;
            } else if (p > 3) {
                return LIKELY;
            } else {
                return UNKNOWN;
            }
        }
    }

    public static double getResponse(SprlClassLabel label) {
        return getResponse(label.ordinal());
    }

    public static double getResponse(int label) {
        if (label == NA.ordinal() && modelNA) {
            return -1;
        } else if (label == UNLIKELY.ordinal()) {
            return 1;
        } else if (label == LIKELY.ordinal()) {
            return 5;
        } else if (label == UNKNOWN.ordinal()) {
            // make sure that unknown is allowed
            if (splitMode == SplitMode.Split_123_45 || splitMode == SplitMode.Split_1234_5) {
                log.warn("getting response for UNKNOWN but split mode doesn't have a slot for UNKNOWN");
            }
            return 3;
        } else {
            if (label == NA.ordinal() && !modelNA) {
                log.warn("getting response for NA but not modeling NA");
            }
            // includes NA and NOT_AN_ARG
            return 0;
        }
    }

}