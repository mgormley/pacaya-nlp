package edu.jhu.nlp.eval;

import java.util.Arrays;
import java.util.Set;
import java.util.TreeSet;

import edu.jhu.nlp.Evaluator;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.data.simple.AnnoSentenceCollection;
import edu.jhu.nlp.sprl.ConfusionMap;
import edu.jhu.nlp.sprl.ConfusionMatrix;
import edu.jhu.nlp.sprl.SprlLabelConverter;
import edu.jhu.nlp.sprl.SprlProperties;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.RoleStructure;
import edu.jhu.pacaya.util.report.Reporter;
import edu.jhu.prim.tuple.Pair;

/**
 * (Fork of F1 evaluator) Computes the precision, recall, and micro-averaged F1.
 * 
 */
public class SprlEvaluator implements Evaluator {

//    private static final Logger log = LoggerFactory.getLogger(SprlEvaluator.class);
    private static final Reporter rep = Reporter.getReporter(SprlEvaluator.class);
    public final static Set<String> nilLabels = new TreeSet<String>(Arrays.asList(SprlLabelConverter.NOT_AN_ARG, SprlLabelConverter.NA, SprlLabelConverter.UNLIKELY));
    private ConfusionMap<String, String> cms;
    private boolean reportIndividual;
    private boolean reportMajorityBaseline;
    private int numSentences = 0;
    private RoleStructure rS;
    private boolean allowSelfLoops;

    public SprlEvaluator(RoleStructure rS, boolean allowSelfLoops, boolean reportIndividual, boolean reportMajorityBaseline, boolean reportOther) {
        this.reportIndividual = reportIndividual;
        this.reportMajorityBaseline = reportMajorityBaseline;
        this.rS = rS;
        this.allowSelfLoops = allowSelfLoops;
        reset();
    }
    
    public void reset() {
        cms = new ConfusionMap<>(nilLabels);
    }
    
    public void report(String dataName) {
        rep.report(dataName+"SPRLnumSentences", ""+numSentences);
        if (reportIndividual) {
            for (String category : cms.getCategories()) {
                report(dataName+"SPRL-"+category, cms.getConfusionMatrix(category), false);
            }
        }
        report(dataName+"SPRL", cms.getTotal(), true);
    }

    public void report(String name, ConfusionMatrix<String> cm, boolean isTotal) {
        cm.reportSummary(name, rep);
        if (reportMajorityBaseline) {
            if (isTotal) { 
                cms.reportClassSpecificMajorityBaseline(name, rep);
            } else {
                cm.reportMajorityBaseline(name, rep);
            }
        }
    }
    
    /** Computes the precision, recall, and micro-averaged F1 of relations mentions. */
    @Override
    public double evaluate(AnnoSentenceCollection predSents, AnnoSentenceCollection goldSents, String dataName) {
        reset();
        accum(cms, predSents, goldSents, rS, allowSelfLoops);
        report(dataName);
        return -cms.getTotal().f1();
    }

    public static void accum(ConfusionMap<String, String> m, AnnoSentenceCollection predSents, AnnoSentenceCollection goldSents, RoleStructure rS, boolean allowPredArgSelfLoops) {
        for (int i = 0; i < goldSents.size(); i++) {
            AnnoSentence goldSent = goldSents.get(i);
            AnnoSentence predSent = predSents.get(i);
            SprlProperties g = goldSent.getSprl();
            SprlProperties p = predSent.getSprl();
            for (Pair<Integer, Integer> pair : SrlFactorGraphBuilder.getPossibleRolePairs(goldSent, rS, allowPredArgSelfLoops, false)) {
                for (String q : g.getLabeledProperties(pair)) {
                    m.recordPrediction(
                            g.get(pair.get1(), pair.get2(), q),
                            p.get(pair.get1(), pair.get2(), q),
                            q);
                }
            }

        }
    }

}
