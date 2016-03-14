package edu.jhu.nlp.eval;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.Evaluator;
import edu.jhu.nlp.data.Properties;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.sprl.SprlClassLabel;
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
    private Set<SprlClassLabel> nilLabels;
    private Collection<String> propsToScore;

    public SprlEvaluator(RoleStructure rS, boolean selfLoops, Set<SprlClassLabel> nilLabels) {
        this(rS, selfLoops, nilLabels, null);
    }

    
    public SprlEvaluator(RoleStructure rS, boolean selfLoops, Set<SprlClassLabel> nilLabels, Collection<String> propsToScore) {
        roleStructure = rS;
        allowSelfLoops = selfLoops;
        this.propsToScore = propsToScore;
        this.nilLabels = nilLabels;
    }

//    public void setPropsToScore(Collection<String> propsToScore) {
//        this.propsToScore = propsToScore;
//    }
    
    /**
     * Puts together the list of predicate, argument, property triples that should be evaluated
     * Only the properties annotated in the gold sentence are evaluated; if propsToScore is not null,
     * then the evaluated set will be restricted to those as well.
     */
    public List<Triple<Integer, Integer, String>> getExamples(AnnoSentence sent, AnnoSentence gold) {
        List<Triple<Integer, Integer, String>> examples = new ArrayList<>();
        for (Pair<Integer, Integer> e : SrlFactorGraphBuilder.getPossibleRolePairs(gold.size(),
                gold.getKnownSprlPreds(), gold.getSprl().keySet(), gold.getPairsToSkip(), roleStructure, allowSelfLoops)) {
            Properties props = gold.getSprl().get(e);
            if (e != null) {
                Set<String> toScore = props.getMap().keySet();
                if (propsToScore != null) {
                    // make a copy and leave only those that we are scoring  
                    toScore = new HashSet<>(toScore);
                    toScore.retainAll(propsToScore);
                }
                for (String q : toScore) {
                    if (props.getMap().keySet().contains(q)) {
                        examples.add(new Triple<>(e.get1(), e.get2(), q));
                    }
                }
            }
        }
        return examples;
    }

    @Override
    public List<String> getLabels(AnnoSentence sent, AnnoSentence gold) {
        List<String> labels = new ArrayList<>();
        Map<Pair<Integer, Integer>, Properties> predSprl = sent.getSprl();
        // get the labels according to the pred sent, but including
        // all and only those possible according to the gold sentence
        for (Triple<Integer, Integer, String> e : getExamples(sent, gold)) {
            SprlClassLabel label = getLabel(predSprl.get(new Pair<>(e.get1(), e.get2())), e.get3());
            labels.add(label.name());
        }
        return labels;
    }

    private SprlClassLabel getLabel(Properties props, String q) {
        if (props == null) {
            return SprlClassLabel.NOT_AN_ARG;
        } else {
            return props.getLabel(q);
        }
    }

    @Override
    protected boolean isNilLabel(String label) {
        return nilLabels.contains(SprlClassLabel.valueOf(label));
    }

    @Override
    protected String getDataType() {
        return String.format("SPRL%s", propsToScore == null ? "" : propsToScore.toString());
    }

}
