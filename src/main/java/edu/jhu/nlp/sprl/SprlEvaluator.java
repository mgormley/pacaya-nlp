package edu.jhu.nlp.sprl;

import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.Evaluator;
import edu.jhu.nlp.data.Properties;
import edu.jhu.nlp.data.Properties.Property;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.data.simple.AnnoSentenceCollection;
import edu.jhu.pacaya.gm.app.Loss;
import edu.jhu.pacaya.util.report.Reporter;
import edu.jhu.prim.tuple.Pair;

/**
 * Computes the per-token SPRL tagging accuracy.
 */
public class SprlEvaluator implements Loss<AnnoSentence>, Evaluator {

    private static final Logger log = LoggerFactory.getLogger(SprlEvaluator.class);
    private static final Reporter rep = Reporter.getReporter(SprlEvaluator.class);

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
        log.info(String.format("SPRL accuracy on %s: %.4f", dataName, accuracy));    
        rep.report(dataName+"SPRLAccuracy", accuracy);
        return getErrors();
    }

    private void evaluate(AnnoSentence pred, AnnoSentence gold) {
        Map<Pair<Integer, Integer>, Properties> goldProps = gold.getSprl();
        Map<Pair<Integer, Integer>, Properties> predProps = pred.getSprl();
        if (predProps != null) {
            assert(predProps.size() == goldProps.size());
        }
        for (Pair<Integer, Integer> pair : goldProps.keySet()) {
            Map<String, Double> gM = goldProps.get(pair).toMap();
            Map<String, Double> pM = predProps.get(pair).toMap();
            for (Property q : Property.values()) {
                if (Math.abs(gM.get(q.name()) - pM.get(q.name())) < 1E-4) {
                    correct++;
                }
                total++;
            }
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
