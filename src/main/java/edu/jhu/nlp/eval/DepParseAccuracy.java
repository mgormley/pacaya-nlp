package edu.jhu.nlp.eval;

import java.util.List;
import java.util.regex.Pattern;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.Evaluator;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.data.simple.AnnoSentenceCollection;
import edu.jhu.pacaya.gm.app.Loss;
import edu.jhu.pacaya.util.report.Reporter;

/**
 * Computes the unlabeled directed dependency accuracy. This is simply the
 * overall fraction of correctly predicted dependencies.
 * 
 * @author mgormley
 */
public class DepParseAccuracy implements Loss<AnnoSentence>, Evaluator {

    /** Regex for matching words consisting of entirely Unicode punctuation characters. */
    static final Pattern PUNCT_RE = Pattern.compile("^\\p{Punct}+$", Pattern.UNICODE_CHARACTER_CLASS);
    private static final Logger log = LoggerFactory.getLogger(DepParseAccuracy.class);
    private static final Reporter rep = Reporter.getReporter(DepParseAccuracy.class);

    private double accuracy;
    private int correct;
    private int total;
    private boolean skipPunctuation;
    private boolean labeled;

    public DepParseAccuracy(boolean skipPunctuation) {
        this.skipPunctuation = skipPunctuation;
    }

    public DepParseAccuracy(boolean skipPunctuation, boolean labeled) {
        this.skipPunctuation = skipPunctuation;
        this.labeled = labeled;
    }
    
    
    /** Gets the number of incorrect dependencies. */
    @Override
    public double loss(AnnoSentence pred, AnnoSentence gold) {
        reset();
        evaluate(pred, gold);
        return getErrors();
    }

    private void reset() {
        correct = 0;
        total = 0;
        accuracy = 0;
    }

    /** Computes the number of correct dependencies, total dependencies, and accuracy. */
    public double evaluate(AnnoSentenceCollection predSents, AnnoSentenceCollection goldSents, String dataName) {
        reset();
        assert(predSents.size() == goldSents.size());
        for (int i = 0; i < goldSents.size(); i++) {
            AnnoSentence gold = goldSents.get(i);
            AnnoSentence pred = predSents.get(i);
            evaluate(pred, gold);
        }
        accuracy = (double) correct / (double) total;
        if (labeled) {
            log.info(String.format("Labeled attachment score on %s: %.4f", dataName, accuracy));        
            rep.report(dataName+"LAS", accuracy);
        } else {
            log.info(String.format("Unlabeled attachment score on %s: %.4f", dataName, accuracy));        
            rep.report(dataName+"UAS", accuracy);
        }
        return getErrors();
    }

    private void evaluate(AnnoSentence pred, AnnoSentence gold) {
        int[] goldParents = gold.getParents();
        int[] predParents = pred.getParents();
        List<String> goldLabels = gold.getDeprels();
        List<String> predLabels = pred.getDeprels();
        if (predParents != null) {
            assert(predParents.length == goldParents.length);
        }
        if (labeled && predLabels != null) {
            assert(predLabels.size() == goldLabels.size());
        }
        for (int c = 0; c < goldParents.length; c++) {
            if (skipPunctuation && isPunctuation(gold.getWord(c))) {
                // Don't score punctuation.
                continue;
            }
            if (labeled) {
                if (predParents != null && predLabels != null 
                        && goldParents[c] == predParents[c] 
                        && goldLabels.get(c).equals(predLabels.get(c))) {
                    correct++;
                }
            } else {
                if (predParents != null && goldParents[c] == predParents[c]) {
                    correct++;
                }
            }
            total++;
        }
    }
    
    public static boolean isPunctuation(String word) {
        return PUNCT_RE.matcher(word).matches();
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
