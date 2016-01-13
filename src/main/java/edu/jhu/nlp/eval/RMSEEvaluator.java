package edu.jhu.nlp.eval;

import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.Evaluator;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.data.simple.AnnoSentenceCollection;
import edu.jhu.pacaya.util.report.Reporter;

/**
 * Computes the precision, recall, and micro-averaged F1.
 * 
 * @author mgormley
 */
public abstract class RMSEEvaluator implements Evaluator {

    private static final Logger log = LoggerFactory.getLogger(RMSEEvaluator.class);
    private static final Reporter rep = Reporter.getReporter(RMSEEvaluator.class);

    private int numStructures; // number of sentences not missing
    private int numStructuresMissing; // number of sentences not annotated
    private int numExamplesNotMissing; // num examples represented in the mean error
    private int numExamplesMissing; // num examples that had mismatch between gold and predicted nils
    private double meanSquaredError;
    private double rootMeanSquaredError;

    /**
     * Returns the predictions for a given sentence.
     */
    protected abstract List<Double> getValues(AnnoSentence sent, AnnoSentence gold);

    /** Gets the type of data, which is used as a prefix for reporting. */
    protected abstract String getDataType();

    protected void reset() {
        numStructures = 0;
        numStructuresMissing = 0;
        numExamplesNotMissing = 0;
        numExamplesMissing = 0;
        meanSquaredError = 0.0;
        rootMeanSquaredError = 0.0;
    }

    /**
     * Computes the precision, recall, and micro-averaged F1 of relations
     * mentions.
     */
    public double evaluate(AnnoSentenceCollection predSents, AnnoSentenceCollection goldSents, String dataName) {
        accum(predSents, goldSents);
        String dataType = getDataType();
        log.info(String.format("Num sents %s: %d", dataName, numStructures));
        log.info(String.format("Num sents not annotated on %s: %d", dataName, numStructuresMissing));
        log.info(String.format("RMSE on %s: %.4f", dataName, rootMeanSquaredError));
        log.info(String.format("Num instances on %s: %d", dataName, numExamplesNotMissing + numExamplesMissing));
        log.info(String.format("Num instances missing on %s: %d", dataName, numExamplesMissing));        
        rep.report(dataName + dataType + "RMSE", rootMeanSquaredError);

        return rootMeanSquaredError;
    }

    /**
     * Computes the precision, recall, and micro-averaged F1 over all the
     * sentences.
     */
    public void accum(AnnoSentenceCollection predSents, AnnoSentenceCollection goldSents) {
        reset();

        assert predSents.size() == goldSents.size();

        // For each sentence.
        for (int s = 0; s < goldSents.size(); s++) {
            AnnoSentence goldSent = goldSents.get(s);
            AnnoSentence predSent = predSents.get(s);
            List<Double> gold = getValues(goldSent, goldSent);
            List<Double> pred = getValues(predSent, goldSent);
            accum(gold, pred);
        }
    }

    /** Accumulate the sufficient statistics for the sentence. */
    public void accum(List<Double> gold, List<Double> pred) {
        numStructures++;
        if (gold == null) {
            return;
        }
        if (pred == null) {
            numStructuresMissing++;
        }
        if (pred != null) {
            assert gold.size() == pred.size();
        }

        // For each pair of named entities.
        for (int k = 0; k < gold.size(); k++) {
            Double goldLabel = gold.get(k);
            Double predLabel = pred.get(k);
            if (goldLabel == null || predLabel == null) {
                numExamplesMissing += 1;
            } else {
                double diff = (goldLabel - predLabel);
                double squaredError = diff * diff;
                if (numExamplesNotMissing == 0) {
                    meanSquaredError = squaredError;
                } else {
                    // meanSquaredError = (meanSquaredError * numExamples +
                    // squaredError) / (numExamples + 1);
                    meanSquaredError *= (double) numExamplesNotMissing / (numExamplesNotMissing + 1);
                    meanSquaredError += squaredError / (numExamplesNotMissing + 1);
                }
                numExamplesNotMissing += 1;
            }
        }
        rootMeanSquaredError = Math.sqrt(meanSquaredError);
    }

    public double getRMSE() {
        return rootMeanSquaredError;
    }

    public int getNumExamples() {
        return numExamplesNotMissing + numExamplesMissing;
    }
    
    public int getNumExamplesMissing() {
        return numExamplesMissing;
    }

    public int getNumStructures() {
        return numStructures;
    }
    
    public int getNumStructuresMissing() {
        return numStructuresMissing;
    }
    
}
