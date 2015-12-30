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
 * Computes the precision, recall, and micro-averaged F1 for named entity recognition.
 * 
 * @author mgormley
 */
public class SprlEvaluator extends F1Evaluator implements Evaluator {

    private static final Logger log = LoggerFactory.getLogger(SprlEvaluator.class);
    private static final Reporter rep = Reporter.getReporter(SprlEvaluator.class);
    private RoleStructure roleStructure = null;
    private boolean allowSelfLoops;
    
    public SprlEvaluator(RoleStructure rS, boolean selfLoops) {
        roleStructure = rS;
        allowSelfLoops = selfLoops;
    }
    
    @Override
    protected List<String> getLabels(AnnoSentence sent) {
        List<String> labels = new ArrayList<>();
        Map<Pair<Integer, Integer>, Properties> sprl = sent.getSprl(); 
        for (Pair<Integer, Integer> e : SrlFactorGraphBuilder.getPossibleRolePairs(sent,  roleStructure,  allowSelfLoops)) {
            Properties props = sprl.get(e);
            double[] propMap = (props != null) ? props.toArray() : null; 
            for (Property q : Property.values()) {
                if (propMap == null) {
                    labels.add(SprlClassLabel.NOT_AN_ARG.name());    
                } else {
                    SprlClassLabel label = SprlClassLabel.getLabel(propMap[q.ordinal()]);
                    labels.add(label.name());
                }
                
            }
        }
        return labels;
    }

    @Override
    protected boolean isNilLabel(String label) {
        return SprlClassLabel.NOT_AN_ARG.name().equals(label);
    }
    
    @Override
    protected String getDataType() {
        return "SPRL";
    }
    
}
