package edu.jhu.nlp.eval;

import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.Evaluator;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.data.simple.AnnoSentenceCollection;
import edu.jhu.pacaya.util.report.Reporter;
import edu.jhu.prim.tuple.Pair;

/**
 * Computes the micro-averaged precision, recall, and F1.
 *
 * @author mgormley
 */
public abstract class F1Evaluator implements Evaluator {

    private static final Logger log = LoggerFactory.getLogger(F1Evaluator.class);
    private static final Reporter rep = Reporter.getReporter(F1Evaluator.class);

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

    /** Gets the type of data, which is used as a prefix for reporting. */
    protected abstract String getDataType();

    /** True iff the label corresponds to the "nil" label. */
    protected abstract boolean isNilLabel(String label);

    /** Returns the labels for a given sentence, or null if no labels are present. */
    protected abstract List<String> getLabels(AnnoSentence sent);

    /** Returns the labels for the gold / predicted sentences, or null if no labels are present. */
    protected Pair<List<String>,List<String>> getLabels(AnnoSentence goldSent, AnnoSentence predSent) {
        return new Pair<>(getLabels(goldSent), getLabels(predSent));
    }

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

        log.debug(String.format("%s # correct positives on %s: %d", dataType, dataName, numCorrectPositive));
        log.debug(String.format("%s # predicted positives on %s: %d", dataType, dataName, numPredictPositive));
        log.debug(String.format("%s # true positives on %s: %d", dataType, dataName, numTruePositive));

        log.info(String.format("%s # sents not annotated on %s: %d", dataType, dataName, numMissing));
        log.info(String.format("%s # instances on %s: %d", dataType, dataName, numInstances));

        // This is not simply tag accuracy for NER and so shouldn't be included.
        //log.info(String.format("%s Accuracy on %s: %.4f", dataType, dataName, (double)(numCorrectPositive + numCorrectNegative)/numInstances));
        
        log.info(String.format("%s Precision on %s: %.4f", dataType, dataName, precision));
        log.info(String.format("%s Recall on %s: %.4f", dataType, dataName, recall));
        log.info(String.format("%s F1 on %s: %.4f", dataType, dataName, f1));

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
            accum(goldSent, predSent);
        }
    }

    /** Accumulate the sufficient statistics for the sentence. */
    public void accum(AnnoSentence goldSent, AnnoSentence predSent) {
        Pair<List<String>, List<String>> pair = getLabels(goldSent, predSent);
        accum(pair.get1(), pair.get2());
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

            boolean goldIsNil = isNilLabel(goldLabel);
            boolean predIsNil = (pred == null) ? false : isNilLabel(predLabel);

            if (pred != null && ((goldIsNil && predIsNil) || (goldLabel != null && goldLabel.equals(predLabel)))){
                if (!goldIsNil) {
                    numCorrectPositive++;
                } else {
                    numCorrectNegative++;
                }
            }
            if (!goldIsNil) {
                numTruePositive++;
            }
            if (pred != null && !predIsNil) {
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
