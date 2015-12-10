package edu.jhu.nlp.eval;

import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.Evaluator;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.data.simple.AnnoSentenceCollection;
import edu.jhu.nlp.relations.RelationMunger;
import edu.jhu.pacaya.util.report.Reporter;

/**
 * Computes the precision, recall, and micro-averaged F1 for named entity recognition.
 * 
 * @author mgormley
 */
public class NerEvaluator extends F1Evaluator implements Evaluator {

    private static final Logger log = LoggerFactory.getLogger(NerEvaluator.class);
    private static final Reporter rep = Reporter.getReporter(NerEvaluator.class);

    // The string which is treated as "no entity". 
    private String outsideStr;
    
    public NerEvaluator() {
        this("O");
    }
    
    public NerEvaluator(String outsideStr) {
        this.outsideStr = outsideStr;
    }

    @Override
    protected List<String> getLabels(AnnoSentence sent) {
        // TODO: Switch from chunks to neTags.
        return sent.getChunks();
    }

    @Override
    protected boolean isNilLabel(String label) {
        if (outsideStr == null) {
            return outsideStr == label;
        } else {
            return outsideStr.equals(label);
        }
    }
    
    @Override
    protected String getDataType() {
        return "NER";
    }
    
}
