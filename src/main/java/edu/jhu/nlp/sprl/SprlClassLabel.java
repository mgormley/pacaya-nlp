package edu.jhu.nlp.sprl;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

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
    
    @Opt(hasArg = true, description = "Comma separated list of sprl labels to consider as nils in computing precision and recall")
    public static String sprlNils = "NOT_AN_ARG";

    // add the labels as string names
    private static ArrayList<String> sprlLabels = null;
    
    public static List<String> getLabels() {
        if (sprlLabels == null) {
            sprlLabels = new ArrayList<>();
            for (SprlClassLabel label : values()) {
                sprlLabels.add(label.name());
            }
        }
        return sprlLabels;
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
        if (label == NA && modelNA) {
            return -1;
        } else if (label == UNLIKELY) {
            return 1;
        } else if (label == LIKELY) {
            return 5;
        } else if (label == UNKNOWN) {
            // make sure that unknown is allowed
            if (splitMode == SplitMode.Split_123_45 || splitMode == SplitMode.Split_1234_5) {
                // this happens especially at the beginning of learning before the fact that
                // this is never observed influences the parameters
                log.debug("getting response for UNKNOWN but split mode doesn't have a slot for UNKNOWN");
            }
            return 3;
        } else {
            if (label == NA && !modelNA) {
                log.debug("getting response for NA but not modeling NA");
            }
            // includes NA and NOT_AN_ARG
            return 0;
        }
    }
    
    public static Set<SprlClassLabel> getNils() {
        String[] splits = sprlNils.split(",");
        HashSet<SprlClassLabel> nils = new HashSet<>();
        for (String s : splits) {
            nils.add(SprlClassLabel.valueOf(s));
        }
        return nils;
    }

}