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
 * (Fork of F1 evaluator)
 * Computes the precision, recall, and micro-averaged F1.
 * 
 */
public class SprlRMSEEvaluator extends RMSEEvaluator implements Evaluator {

    private static final Logger log = LoggerFactory.getLogger(SprlRMSEEvaluator.class);
    private static final Reporter rep = Reporter.getReporter(SprlRMSEEvaluator.class);
    private RoleStructure roleStructure = null;
    private boolean allowSelfLoops;
    private boolean excludeNils;
    private Property propToScore = null;

    public SprlRMSEEvaluator(RoleStructure rS, boolean selfLoops, boolean excludeNils) {
        this(rS, selfLoops, excludeNils, null);
    }
    
    /**
     * if propToScore is not null, then only that property will be evaluated 
     */
    public SprlRMSEEvaluator(RoleStructure rS, boolean selfLoops, boolean excludeNils, Property propToScore) {
        roleStructure = rS;
        allowSelfLoops = selfLoops;
        this.excludeNils = excludeNils;
        this.propToScore = propToScore;
    }
    
    @Override
    protected List<Double> getValues(AnnoSentence sent, AnnoSentence gold) {
        List<Double> values = new ArrayList<>();
        Map<Pair<Integer, Integer>, Properties> sprl = sent.getSprl(); 
        // get the labels according to the pred sent, but including
        // all and only those possible according to the gold sentence
        for (Pair<Integer, Integer> e : SrlFactorGraphBuilder.getPossibleRolePairs(gold.size(),
                gold.getSprlPreds(), gold.getSprl().keySet(), roleStructure,  allowSelfLoops)) {
            Properties props = sprl.get(e);
            double[] propMap = (props != null) ? props.toArray() : null; 
            if (propToScore != null) {
                addLabel(values, propMap, propToScore);
            } else {
                for (Property q : Property.values()) {
                    addLabel(values, propMap, q);
                }
            }
        }
        return values;
    }

    private void addLabel(List<Double> values, double[] propMap, Property q) {
        if (propMap == null) {
            if (excludeNils) {
                // if we exclude nils, then we will just pass null here
                values.add(null);
            } else {
                values.add(0.0);
            }
        } else {
            values.add((propMap[q.ordinal()] - 1) / 4.0); 
        }
    }

    @Override
    protected String getDataType() {
        return String.format("SPRL-RMSE%s%s", propToScore == null ? "" : ("[" + propToScore.name() + "]"), excludeNils ? "(skippingNils)" : "");
    }
    
}
