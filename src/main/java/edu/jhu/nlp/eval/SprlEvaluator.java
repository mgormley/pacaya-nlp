package edu.jhu.nlp.eval;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.Evaluator;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.sprl.BinarySprlLabelConverter;
import edu.jhu.nlp.sprl.SprlLabelConverter;
import edu.jhu.nlp.sprl.SprlProperties;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.RoleStructure;
import edu.jhu.pacaya.util.report.Reporter;
import edu.jhu.prim.tuple.Pair;
import edu.jhu.prim.tuple.Triple;

/**
 * (Fork of F1 evaluator) Computes the precision, recall, and micro-averaged F1.
 * 
 */
public class SprlEvaluator extends LabelEvaluator implements Evaluator {

    private static final Logger log = LoggerFactory.getLogger(SprlEvaluator.class);
    private static final Reporter rep = Reporter.getReporter(SprlEvaluator.class);
    private RoleStructure roleStructure = null;
    private boolean allowSelfLoops;
    public final static Set<String> nilLabels = new TreeSet<String>(Arrays.asList(SprlLabelConverter.NOT_AN_ARG, SprlLabelConverter.NA, SprlLabelConverter.UNLIKELY));
    private Collection<String> propsToScore;

    public SprlEvaluator(RoleStructure rS, boolean selfLoops) {
        this(rS, selfLoops, null);
    }

    
    public SprlEvaluator(RoleStructure rS, boolean selfLoops, Collection<String> propsToScore) {
        roleStructure = rS;
        allowSelfLoops = selfLoops;
        this.propsToScore = propsToScore;
    }

    /**
     * Returns a list of only those pred-arg pairs that should be evaluated
     */
    public List<Pair<Integer, Integer>> getExamplePairs(AnnoSentence sent, AnnoSentence gold) {
        return SrlFactorGraphBuilder.getPossibleRolePairs(gold.size(),
                gold.getKnownSprlPreds(), gold.getSprl().getPairs(), gold.getPairsToSkip(), roleStructure, allowSelfLoops);
    }

    @Override
    public List<String> getLabels(AnnoSentence sent, AnnoSentence gold) {
        List<String> labels = new ArrayList<>();
        SprlProperties predSprl = sent.getSprl();
        // get the labels according to the pred sent, but including
        // all and only those possible according to the gold sentence
        for (Triple<Integer, Integer, String> e : gold.getSprl().getLabeledProperties()) {
            labels.add(predSprl.get(e));
        }
        return labels;
    }

    @Override
    protected Set<String> getNilLabels() {
        return nilLabels;
    }

    @Override
    protected String getDataType() {
        return String.format("SPRL%s", propsToScore == null ? "" : propsToScore.toString());
    }

}
