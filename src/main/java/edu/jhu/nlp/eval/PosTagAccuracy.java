package edu.jhu.nlp.eval;

import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.Evaluator;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.data.simple.AnnoSentenceCollection;
import edu.jhu.pacaya.gm.app.Loss;
import edu.jhu.pacaya.util.report.Reporter;

/**
 * Computes the per-token POS tagging accuracy.
 * 
 * @author mgormley
 */
public class PosTagAccuracy implements Loss<AnnoSentence>, Evaluator {

    private static final Logger log = LoggerFactory.getLogger(PosTagAccuracy.class);
    private static final Reporter rep = Reporter.getReporter(PosTagAccuracy.class);

    private double accuracy;
    private int correct;
    private int total;
    
    /** Gets the number of incorrect tags. */
    @Override
    public double loss(AnnoSentence pred, AnnoSentence gold) {
        correct = 0;
        total = 0;
        evaluate(pred, gold);
        return getErrors();
    }

    /** Computes the number of correct tags, total tags, and accuracy. */
    public double evaluate(AnnoSentenceCollection predSents, AnnoSentenceCollection goldSents, String dataName) {
        correct = 0;
        total = 0;
        assert(predSents.size() == goldSents.size());
        for (int i = 0; i < goldSents.size(); i++) {
            AnnoSentence gold = goldSents.get(i);
            AnnoSentence pred = predSents.get(i);
            evaluate(pred, gold);
        }
        accuracy = (double) correct / (double) total;
        log.info(String.format("POS tag accuracy on %s: %.4f", dataName, accuracy));    
        rep.report(dataName+"PosAccuracy", accuracy);
        return getErrors();
    }

    private void evaluate(AnnoSentence pred, AnnoSentence gold) {
        List<String> goldTags = gold.getPosTags();
        List<String> predTags  = pred.getPosTags();
        if (predTags != null) {
            assert(predTags.size() == goldTags.size());
        }
        for (int i = 0; i < goldTags.size(); i++) {
            if (predTags != null) {
                if (goldTags.get(i).equals(predTags.get(i))) {
                    correct++;
                }
            }
            total++;            
        }
    }
    
    public double getAccuracy() {
        return accuracy;
    }

    public int getCorrect() {
        return correct;
    }

    public int getTotal() {
        return total;
    }

    public double getErrors() {
        return total - correct;
    }

}
