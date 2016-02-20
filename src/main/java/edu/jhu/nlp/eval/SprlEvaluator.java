package edu.jhu.nlp.eval;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

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
    private Set<SprlClassLabel> nilLabels;

    /**
     * Evaluator for all properties
     */
    public SprlEvaluator(RoleStructure rS, boolean selfLoops, Set<SprlClassLabel> nilLabels) {
        this(rS, selfLoops, nilLabels, null);
    }

    public SprlEvaluator(RoleStructure rS, boolean selfLoops, Set<SprlClassLabel> nilLabels, Property propertyToScore) {
        roleStructure = rS;
        allowSelfLoops = selfLoops;
        this.propToScore = propertyToScore;
        this.nilLabels = nilLabels;
    }

    @Override
    public List<String> getLabels(AnnoSentence sent, AnnoSentence gold) {
        List<String> labels = new ArrayList<>();
        Map<Pair<Integer, Integer>, Properties> sprl = sent.getSprl();
        // get the labels according to the pred sent, but including
        // all and only those possible according to the gold sentence
        for (Pair<Integer, Integer> e : SrlFactorGraphBuilder.getPossibleRolePairs(gold.size(),
                gold.getKnownSprlPreds(), gold.getSprl().keySet(), roleStructure, allowSelfLoops)) {
            Properties props = sprl.get(e);
            List<SprlClassLabel> propList = (props != null) ? props.toLabels() : null;
            if (propToScore != null) {
                addLabel(labels, propList, propToScore);
            } else {
                for (Property q : Property.values()) {
                    addLabel(labels, propList, q);
                }
            }
        }
        return labels;
    }

    private void addLabel(List<String> labels, List<SprlClassLabel> propList, Property q) {
        if (propList == null) {
            labels.add(SprlClassLabel.NOT_AN_ARG.name());
        } else {
            labels.add(propList.get(q.ordinal()).name());
        }
    }

    @Override
    protected boolean isNilLabel(String label) {
        return nilLabels.contains(SprlClassLabel.valueOf(label));
    }

    @Override
    protected String getDataType() {
        return String.format("SPRL%s", propToScore == null ? "" : ("[" + propToScore.name() + "]"));
    }

}
