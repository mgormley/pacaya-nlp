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
import edu.jhu.prim.tuple.Triple;

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

    public List<Triple<Integer, Integer, Property>> getExamples(AnnoSentence sent, AnnoSentence gold) {
        List<Triple<Integer, Integer, Property>> examples = new ArrayList<>();
        for (Pair<Integer, Integer> e : SrlFactorGraphBuilder.getPossibleRolePairs(gold.size(),
                gold.getKnownSprlPreds(), gold.getSprl().keySet(), gold.getPairsToSkip(), roleStructure, allowSelfLoops)) {
            if (propToScore != null) {
                examples.add(new Triple<>(e.get1(), e.get2(), propToScore));
            } else {
                for (Property q : Property.values()) {
                    examples.add(new Triple<>(e.get1(), e.get2(), q));
                }
            }
        }
        return examples;
    }

    @Override
    public List<String> getLabels(AnnoSentence sent, AnnoSentence gold) {
        List<String> labels = new ArrayList<>();
        Map<Pair<Integer, Integer>, Properties> sprl = sent.getSprl();
        // get the labels according to the pred sent, but including
        // all and only those possible according to the gold sentence
        for (Triple<Integer, Integer, Property> e : getExamples(sent, gold)) {
            SprlClassLabel label = getLabel(sprl.get(new Pair<>(e.get1(), e.get2())), e.get3());
            labels.add(label.name());
        }
        return labels;
    }

    private SprlClassLabel getLabel(Properties props, Property q) {
        if (props == null) {
            return SprlClassLabel.NOT_AN_ARG;
        } else {
            return props.getLabel(q.name());
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
