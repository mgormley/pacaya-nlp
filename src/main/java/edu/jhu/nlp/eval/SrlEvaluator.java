package edu.jhu.nlp.eval;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.Evaluator;
import edu.jhu.nlp.data.DepGraph;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.pacaya.util.Prm;
import edu.jhu.pacaya.util.report.Reporter;
import edu.jhu.prim.tuple.Pair;

/**
 * Computes the micro-averaged precision, recall, and F1 of SRL.
 * 
 * @author mgormley
 */
// TODO: Support other options: predictSense = true, predictPredicatePosition = true.
public class SrlEvaluator extends F1Evaluator implements Evaluator {

    public static class SrlEvaluatorPrm extends Prm {
        private static final long serialVersionUID = 1L;
        /** Whether to do labeled or unlabeled evaluation. */
        public boolean labeled = true;
        /** Whether to evaluate predicate sense. */
        public boolean evalPredSense = true;
        /** Whether to evaluate predicate position. */
        public boolean evalPredPosition = false;
        /** Whether to evaluate arguments (i.e. semantic roles). */
        public boolean evalRoles = true;
        /** Mimic eval09.pl definition of predicate sense equality. */
        public boolean mimicEval09pl = true;
        public boolean evalAmbiguousSrlLabels = false;
        public SrlEvaluatorPrm() { }
        public SrlEvaluatorPrm(boolean labeled, boolean evalSense, boolean evalPredicatePosition, boolean evalRoles) {
            this.labeled = labeled;
            this.evalPredSense = evalSense;
            this.evalPredPosition = evalPredicatePosition;
            this.evalRoles = evalRoles;
        }
    }
    
    private static final Logger log = LoggerFactory.getLogger(SrlEvaluator.class);
    private static final Reporter rep = Reporter.getReporter(SrlEvaluator.class);
    private static final String NO_LABEL = "__NO_LABEL__";
    private static final String SOME_LABEL = "__SOME_LABEL__";

    private final SrlEvaluatorPrm prm;

    public SrlEvaluator(SrlEvaluatorPrm prm) {
        this.prm = prm;
    }

    @Override
    protected String getDataType() {
        String detail = "Srl";
        detail += prm.labeled ? "Labeled" : "Unlabeled";
        detail += prm.evalPredSense ? "Sense" : "";
        detail += prm.evalPredPosition ? "Position" : "";
        detail += prm.evalRoles ? "Roles" : "";
        return detail;
    }

    @Override
    protected boolean isNilLabel(String label) {
        return NO_LABEL.equals(label);
    }

    @Override
    protected Pair<List<String>,List<String>> getLabels(AnnoSentence goldSent, AnnoSentence predSent) {
        DepGraph gold = goldSent.getSrlGraph();
        DepGraph pred = predSent.getSrlGraph();
        
        List<String> goldLabels = new ArrayList<>();
        List<String> predLabels = new ArrayList<>();
        
        if (gold == null) { return new Pair<>(null, null); }
        if (pred == null) { predLabels = null; }
        
        // For each gold edge.
        int n = goldSent.size();
        for (int p=-1; p < n; p++) {          
            if (!prm.evalPredSense && !prm.evalPredPosition && p == -1) {
                // Exclude arcs from the virtual root to predicates.
                continue;
            }
            if (!prm.evalRoles && p != -1) {
                // Only consider arcs from the virtual root.
                continue;
            }
            for (int c=0; c < n; c++) {                      
                // in some cases, some pairs that are known don't have labels that we can evaluate on; skip those
                if (goldSent.getPairsToSkip().contains(new Pair<>(p, c))) {
                    continue;
                }
                if (!prm.evalPredPosition && !hasPredicateForEdge(gold, p, c)) {
                    // Only consider predicates which appear in the gold annotations.
                    continue;
                }
                goldLabels.add(getLabel(gold, p, c));
                if (pred != null) {
                    predLabels.add(getLabel(pred, p, c));
                }
            }
        }
        
        return new Pair<>(goldLabels, predLabels);
    }
    
    private boolean hasPredicateForEdge(DepGraph gold, int p, int c) {
        if (p == -1 && gold.get(p, c) == null) {
            // The parent for this edge is a predicate not found in the gold data.
            return false;
        } else if (p != -1 && gold.get(-1, p) == null) {
            // The child for this edge is a predicate not found in the gold data.
            return false;
        } else {
            // This edge has a predicate in the gold data.
            return true;
        }
    }

    Pattern senseSuffix = Pattern.compile(".+\\.(.+)");
    
    private String getLabel(DepGraph dg, int p, int c) {
        if (dg == null) {
            return null;
        }
        String label = dg.get(p, c);
        if (label == null) {
            return NO_LABEL;
        } else if (!prm.labeled || (p == -1 && !prm.evalPredSense)){
            return SOME_LABEL;
        } else if (prm.mimicEval09pl && p == -1) {
            // The eval09.pl script only checks if the numeric sense matches.
            // Thus, trading.01 and trade.01 would count as correct -- we mimic
            // this behavior here.
            Matcher match = senseSuffix.matcher(label);
            if (match.matches()) {
                return match.group(1);
            }
            return label;
        } else {
            return label;
        }
    }

    @Override
    protected List<String> getLabels(AnnoSentence sent) {
        throw new IllegalStateException("This method is never called.");
    }
    
}

