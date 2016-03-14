package edu.jhu.nlp.eval;

import java.util.List;
import java.util.Set;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.Evaluator;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.data.simple.AnnoSentenceCollection;
import edu.jhu.nlp.sprl.ConfusionMatrix;
import edu.jhu.pacaya.util.report.Reporter;

/**
 * Computes the precision, recall, and micro-averaged F1.
 * 
 * @author mgormley
 */
public abstract class LabelEvaluator implements Evaluator {

    private static final Logger log = LoggerFactory.getLogger(LabelEvaluator.class);
    private static final Reporter rep = Reporter.getReporter(LabelEvaluator.class);

    private ConfusionMatrix<String> cm;
    private int numMissing;
    
    /** Returns the labels for a given sentence. Takes the gold
     * sentence incase the evaluator need gold information to
     * determine the order of the labels for example. (returning a map
     * instead might solve this problem)
     */
    protected abstract List<String> getLabels(AnnoSentence sent, AnnoSentence gold);

    /** True iff the label corresponds to the "nil" label. */
    protected abstract Set<String> getNilLabels();

    /** Gets the type of data, which is used as a prefix for reporting. */
    protected abstract String getDataType();

    protected void reset() {
        cm  = new ConfusionMatrix<>(getNilLabels());
        numMissing = 0;
    }
    
    /** Computes the precision, recall, and micro-averaged F1 of relations mentions. */
    public double evaluate(AnnoSentenceCollection predSents, AnnoSentenceCollection goldSents, String dataName) {
        accum(predSents, goldSents);
        
        String dataType = getDataType();
        log.info(String.format("Num sents not annotated on %s: %d", dataName, numMissing));
        log.info(String.format("Accuracy on %s: %.4f", dataName, (double) getAccuracy()));
        log.info(String.format("Num instances on %s: %d", dataName, cm.getTotal()));
        log.info(String.format("Num true positives on %s: %d", dataName, cm.getCorrectHits()));
        log.info(String.format("Precision on %s: %.4f", dataName, getPrecision()));
        log.info(String.format("Recall on %s: %.4f", dataName, getRecall()));
        log.info(String.format("F1 on %s: %.4f", dataName, getF1()));
        
        rep.report(dataName+dataType+"Accuracy", getAccuracy());
        rep.report(dataName+dataType+"Precision", getPrecision());
        rep.report(dataName+dataType+"Recall", getRecall());
        rep.report(dataName+dataType+"F1", getF1());
        
        return -getF1();
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
            cm.recordPrediction(gold.get(k), pred.get(k));
        }
    }

    public double getAccuracy() {
        return cm.accuracy();
    }

    public double getPrecision() {
        return cm.precision();
    }

    public double getRecall() {
        return cm.recall();
    }

    public double getF1() {
        return cm.f1();
    }

    public int getNumCorrectPositive() {
        return cm.getCorrectHits();
    }

    public int getNumCorrectNegative() {
        return cm.getCorrectNils();
    }

    public int getNumPredictPositive() {
        return cm.getPredictedeHits();
    }

    public int getNumTruePositive() {
        return cm.getPossibleHits();
    }

    public int getNumInstances() {
        return cm.getTotal();
    }

    public int getNumMissing() {
        return numMissing;
    }

    public final static double harmonicMean(double a, double b) {
        return (a == 0.0 && b == 0.0) ? 0.0 : 2 * a * b / (a + b);
    }
    
}
