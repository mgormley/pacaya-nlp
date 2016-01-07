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
 * Computes the precision, recall, and micro-averaged F1.
 * 
 * @author mgormley
 */
public abstract class LabelEvaluator implements Evaluator {

    private static final Logger log = LoggerFactory.getLogger(LabelEvaluator.class);
    private static final Reporter rep = Reporter.getReporter(LabelEvaluator.class);

    private double precision;
    private double recall;
    private double f1;
    // Precision = # correctly predicted positive / # predicted positive
    // Recall = # correctly predicted positive / # true positive
    private int numCorrectPositive;
    private int numCorrectNegative;
    private int numPredictPositive;
    private int numTruePositive;
    private int numInstances;
    private int numMissing;
    
    /** Returns the labels for a given sentence. Takes the gold
     * sentence incase the evaluator need gold information to
     * determine the order of the labels for example. (returning a map
     * instead might solve this problem)
     */
    protected abstract List<String> getLabels(AnnoSentence sent, AnnoSentence gold);

    /** True iff the label corresponds to the "nil" label. */
    protected abstract boolean isNilLabel(String label);

    /** Gets the type of data, which is used as a prefix for reporting. */
    protected abstract String getDataType();

    protected void reset() {
        precision = 0;
        recall = 0;
        f1 = 0;
        numCorrectPositive = 0;
        numCorrectNegative = 0;
        numPredictPositive = 0;
        numTruePositive = 0;
        numInstances = 0;
        numMissing = 0;
    }
    
    /** Computes the precision, recall, and micro-averaged F1 of relations mentions. */
    public double evaluate(AnnoSentenceCollection predSents, AnnoSentenceCollection goldSents, String dataName) {
        accum(predSents, goldSents);
        
        String dataType = getDataType();
        log.info(String.format("Num sents not annotated on %s: %d", dataName, numMissing));
        log.info(String.format("Accuracy on %s: %.4f", dataName, (double)(numCorrectPositive + numCorrectNegative)/numInstances));
        log.info(String.format("Num instances on %s: %d", dataName, numInstances));
        log.info(String.format("Num true positives on %s: %d", dataName, numTruePositive));
        log.info(String.format("Precision on %s: %.4f", dataName, precision));
        log.info(String.format("Recall on %s: %.4f", dataName, recall));
        log.info(String.format("F1 on %s: %.4f", dataName, f1));
        
        rep.report(dataName+dataType+"Precision", precision);
        rep.report(dataName+dataType+"Recall", recall);
        rep.report(dataName+dataType+"F1", f1);
        
        return -f1;
    }
    
    /** Computes the precision, recall, and micro-averaged F1 over all the sentences. */
    public void accum(AnnoSentenceCollection predSents, AnnoSentenceCollection goldSents) {
        reset();
        
        assert predSents.size() == goldSents.size();
        
        // For each sentence.
        for (int s = 0; s < goldSents.size(); s++) {
            AnnoSentence goldSent = goldSents.get(s);
            AnnoSentence predSent = predSents.get(s);
            List<String> gold = getLabels(goldSent, goldSent);
            List<String> pred = getLabels(predSent, goldSent);
            accum(gold, pred);            
        }
    }

    /** Accumulate the sufficient statistics for the sentence. */
    public void accum(List<String> gold, List<String> pred) {
        if (gold == null) { return; }
        if (pred == null) { numMissing++; }
        if (pred != null) { assert gold.size() == pred.size(); }
        
        // For each pair of named entities.
        for (int k=0; k<gold.size(); k++) {                
            String goldLabel = gold.get(k);
            String predLabel = (pred == null) ? null : pred.get(k);
            
            if (goldLabel.equals(predLabel)) {
                if (!isNilLabel(goldLabel)) {
                    numCorrectPositive++;
                } else {
                    numCorrectNegative++;
                }
            }
            if (!isNilLabel(goldLabel)) {
                numTruePositive++;
            }
            if (!isNilLabel(predLabel)) {
                numPredictPositive++;
            }
            numInstances++;
            log.trace(String.format("goldLabel=%s predLabel=%s", goldLabel, predLabel));                    
        }
        precision = numPredictPositive == 0 ? 0.0 : (double) numCorrectPositive / numPredictPositive;
        recall = numTruePositive == 0 ? 0.0 :  (double) numCorrectPositive / numTruePositive;
        f1 = (precision == 0.0 && recall == 0.0) ? 0.0 : (double) (2 * precision * recall) / (precision + recall);
    }

    public double getPrecision() {
        return precision;
    }

    public double getRecall() {
        return recall;
    }

    public double getF1() {
        return f1;
    }

    public int getNumCorrectPositive() {
        return numCorrectPositive;
    }

    public int getNumCorrectNegative() {
        return numCorrectNegative;
    }

    public int getNumPredictPositive() {
        return numPredictPositive;
    }

    public int getNumTruePositive() {
        return numTruePositive;
    }

    public int getNumInstances() {
        return numInstances;
    }

    public int getNumMissing() {
        return numMissing;
    }
    
}
