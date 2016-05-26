package edu.jhu.nlp.eval;

import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.Evaluator;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.relations.RelationMunger;
import edu.jhu.pacaya.util.report.Reporter;

/**
 * Computes the precision, recall, and micro-averaged F1 of relations mentions.
 * 
 * @author mgormley
 */
public class RelationEvaluator extends F1Evaluator implements Evaluator {

    private static final Logger log = LoggerFactory.getLogger(RelationEvaluator.class);
    private static final Reporter rep = Reporter.getReporter(RelationEvaluator.class);

    @Override
    protected String getDataType() {
        return "Rel";
    }

    @Override
    protected boolean isNilLabel(String label) {
        return RelationMunger.isNoRelationLabel(label);
    }

    @Override
    protected List<String> getLabels(AnnoSentence sent) {
        return sent.getRelLabels();
    }
        
}
