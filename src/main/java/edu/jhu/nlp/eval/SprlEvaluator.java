package edu.jhu.nlp.eval;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.Evaluator;
import edu.jhu.nlp.data.Properties;
import edu.jhu.nlp.data.Properties.Property;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.sprl.SprlClassLabel;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.RoleStructure;
import edu.jhu.pacaya.util.report.Reporter;
import edu.jhu.prim.tuple.Pair;

/**
 * (Fork of F1 evaluator) Computes the precision, recall, and micro-averaged F1.
 * 
 */
public class SprlEvaluator extends LabelEvaluator implements Evaluator {

    private static final Logger log = LoggerFactory.getLogger(SprlEvaluator.class);
    private static final Reporter rep = Reporter.getReporter(SprlEvaluator.class);
    private RoleStructure roleStructure = null;
    private Property propToScore = null; // if not null, only include labels
                                         // for the particular property
    private boolean allowSelfLoops;

    /**
     * Evaluator for all properties
     */
    public SprlEvaluator(RoleStructure rS, boolean selfLoops) {
        this(rS, selfLoops, null);
    }

    public SprlEvaluator(RoleStructure rS, boolean selfLoops, Property propertyToScore) {
        roleStructure = rS;
        allowSelfLoops = selfLoops;
        this.propToScore = propertyToScore;
    }

    @Override
    protected List<String> getLabels(AnnoSentence sent, AnnoSentence gold) {
        List<String> labels = new ArrayList<>();
        Map<Pair<Integer, Integer>, Properties> sprl = sent.getSprl();
        // get the labels according to the pred sent, but including
        // all and only those possible according to the gold sentence
        for (Pair<Integer, Integer> e : SrlFactorGraphBuilder.getPossibleRolePairs(gold, roleStructure,
                allowSelfLoops)) {
            Properties props = sprl.get(e);
            double[] propMap = (props != null) ? props.toArray() : null;
            if (propToScore != null) {
                addLabel(labels, propMap, propToScore);
            } else {
                for (Property q : Property.values()) {
                    addLabel(labels, propMap, q);
                }
            }
        }
        return labels;
    }

    private void addLabel(List<String> labels, double[] propMap, Property q) {
        if (propMap == null) {
            labels.add(SprlClassLabel.NOT_AN_ARG.name());
        } else {
            SprlClassLabel label = SprlClassLabel.getLabel(propMap[q.ordinal()]);
            labels.add(label.name());
        }
    }

    @Override
    protected boolean isNilLabel(String label) {
        return SprlClassLabel.NOT_AN_ARG.name().equals(label);
    }

    @Override
    protected String getDataType() {
        return String.format("SPRL[%s]", propToScore == null ? "ALL" : propToScore.name());
    }

}
