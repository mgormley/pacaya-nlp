package edu.jhu.nlp.eval;

import java.io.File;
import java.io.IOException;

import org.apache.commons.cli.ParseException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.data.simple.AnnoSentenceCollection;
import edu.jhu.nlp.data.simple.AnnoSentenceReader;
import edu.jhu.nlp.data.simple.AnnoSentenceReader.AnnoSentenceReaderPrm;
import edu.jhu.nlp.data.simple.AnnoSentenceReader.DatasetType;
import edu.jhu.pacaya.util.cli.ArgParser;
import edu.jhu.pacaya.util.cli.Opt;
import edu.jhu.prim.arrays.DoubleArrays;
import edu.jhu.prim.arrays.IntArrays;
import edu.jhu.prim.util.random.Prng;
import edu.jhu.prim.util.random.RandBits;

/**
 * Significance tests:
 * <ol>
 * <li>Paired permutation test as given by Yeh (2000).</li>
 * <li>Paired bootstrap test as given by Berg-Kirkpatrick & Klein (2012).</li> 
 * </ol>
 * @author mgormley
 */
public class SignificanceTests {

    private static final Logger log = LoggerFactory.getLogger(SignificanceTests.class);

    public static double pptDepParseAccuracy(
            AnnoSentenceCollection goldSents, 
            AnnoSentenceCollection predSents1, 
            AnnoSentenceCollection predSents2, 
            boolean skipPunctuation) {
        final int numSamples = (int) Math.pow(2, 20);
        
        new DepParseAccuracy(skipPunctuation).evaluate(predSents1, goldSents, "pred1");
        new DepParseAccuracy(skipPunctuation).evaluate(predSents2, goldSents, "pred2");
        
        // Compute the scores for each data set.
        double[] scores1 = new double[goldSents.size()];
        double[] scores2 = new double[goldSents.size()];
        for (int i=0; i<goldSents.size(); i++) {
            AnnoSentence gold = goldSents.get(i);
            AnnoSentence pred1 = predSents1.get(i);
            AnnoSentence pred2 = predSents2.get(i);
            DepParseAccuracy acc1 = new DepParseAccuracy(skipPunctuation);
            acc1.loss(pred1, gold);
            DepParseAccuracy acc2 = new DepParseAccuracy(skipPunctuation);
            acc2.loss(pred2, gold);
            scores1[i] = acc1.getCorrect();
            scores2[i] = acc2.getCorrect();
        }
        
        return pairedPermutationTest(scores1, scores2, numSamples);
    }

    public static double bootstrapDepParseAccuracy(
            AnnoSentenceCollection goldSents, 
            AnnoSentenceCollection predSents1, 
            AnnoSentenceCollection predSents2, 
            boolean skipPunctuation) {
        final int numSamples = (int) Math.pow(2, 20);
        
        new DepParseAccuracy(skipPunctuation).evaluate(predSents1, goldSents, "pred1");
        new DepParseAccuracy(skipPunctuation).evaluate(predSents2, goldSents, "pred2");
        
        // Compute the scores for each data set.
        double[][] ss1 = new double[goldSents.size()][2];
        double[][] ss2 = new double[goldSents.size()][2];
        for (int i=0; i<goldSents.size(); i++) {
            AnnoSentence gold = goldSents.get(i);
            AnnoSentence pred1 = predSents1.get(i);
            AnnoSentence pred2 = predSents2.get(i);
            DepParseAccuracy acc1 = new DepParseAccuracy(skipPunctuation);
            acc1.loss(pred1, gold);
            DepParseAccuracy acc2 = new DepParseAccuracy(skipPunctuation);
            acc2.loss(pred2, gold);
            ss1[i][0] = acc1.getCorrect();
            ss1[i][1] = acc1.getTotal();
            ss2[i][0] = acc2.getCorrect();
            ss2[i][1] = acc2.getTotal();
        }
        
        return bootstrapTest(ss1, ss2, numSamples);
    }

    /** Paired permutation test as given by Yeh (2000). */
    public static double pairedPermutationTest(double[] scores1, double[] scores2, final int numSamples) {
        assert scores1.length == scores2.length;
        
        // Since accuracy is proportional to the number of correct labels, 
        // we simply compute the p-value for number-correct.
        double[] diffs = new double[scores1.length];
        for (int i=0; i<scores1.length; i++) {
            diffs[i] = scores1[i] - scores2[i];
        }
        
        double mean = DoubleArrays.mean(diffs);
        log.trace("Score mean: " + mean);
        if (mean < 0) {
            throw new IllegalStateException("we assume pred1 outperformed pred2");
        }
        
        // Shuffle and compute the mean.
        int numGte = 0;
        for (int s=0; s<numSamples; s++) {
            double shufMean = getShuffledMean(diffs);
            if (shufMean >= mean) {
                numGte++;
            }
        }

        // Estimate the p-value.
        // 
        // Yeh (2000) cites (Noreen, 1989, Sec. 3A.3) as defining the estimate of the p-value as below.
        double pval = (numGte + 1.0) / (numSamples + 1.0);
        log.trace("p-value (paired permutation test): " + pval);
        return pval;
    }

    private static double getShuffledMean(double[] diffs) {
        RandBits rand = new RandBits();
        double sum = 0;
        for (int i=0; i<diffs.length; i++) {
            sum += rand.nextBit() ? diffs[i] : -diffs[i];
        }
        return sum / diffs.length;
    }

    /** Paired bootstrap test as given by Berg-Kirkpatrick & Klein (2012). */ 
    public static double bootstrapTest(double[][] ss1, double[][] ss2, final int numSamples) {
        assert ss1.length == ss2.length;
        assert ss1[0].length == ss2[0].length;
        
        int numSents = ss1.length;

        int numGt = 0;
        // Compute the difference in the metric on the true test set.
        int[] sample = IntArrays.range(numSents);
        double diff = computeSampleDiff(ss1, ss2, sample); 
        log.debug("true diff = {}", diff);
        for (int s=0; s<numSamples; s++) {
            // Sample n sentences with replacement.
            for (int ii=0; ii<sample.length; ii++) {
                sample[ii] = Prng.nextInt(numSents);
            }
            // Compute the difference in the metric on the sample.
            double diff_s = computeSampleDiff(ss1, ss2, sample);
            // Update the counter if there's a difference of the two times the true delta.
            if (diff_s > 2*diff) {
                numGt++;
            }
            if (s % 10000 == 0 && s > 0) {
                log.debug("s = {} p = {}", s, 1.0 * numGt / s);
            }
        }
        return 1.0 * numGt / numSamples;
    }

    private static double computeSampleDiff(double[][] ss1, double[][] ss2, int[] sample) {
        final int numSents = ss1.length;
        final int numStats = ss1[0].length;
        // Add the sufficient statistics for the sample.
        double[][] accum = new double[2][numStats];
        for (int ii=0; ii<numSents; ii++) {
            int i = sample[ii];
            for (int d=0; d<numStats; d++) {
                accum[0][d] += ss1[i][d];
                accum[1][d] += ss2[i][d];
            }
        }
        // Compute the metric (accuracy) for each sampled dataset.
        double acc1 = accum[0][0] / accum[0][1];
        double acc2 = accum[1][0] / accum[1][1];
        log.trace("acc1 = {} acc2 = {}", acc1, acc2);
        // Compute the difference in metric.
        double diff = acc1 - acc2;
        return diff;
    }
    
    @Opt(required=true, hasArg=true, description="The path to the gold data")
    public static File gold = null;
    @Opt(required=true, hasArg=true, description="The path to the predicted data (set 1)")
    public static File pred1 = null;
    @Opt(required=true, hasArg=true, description="The path to the predicted data (set 2)")
    public static File pred2 = null;
    @Opt(required=true, hasArg=true, description="The path to the predicted data (set 2)")
    public static DatasetType type = null;
    @Opt(required=true, hasArg=true, description="Whether to skip punctuation (dependency accuracy only)")
    public static boolean skipPunctuation = false;
    
    public static void main(String[] args) throws ParseException, IOException {
        ArgParser parser = new ArgParser(SignificanceTests.class);
        parser.registerClass(SignificanceTests.class);
        parser.parseArgs(args);
                
        AnnoSentenceCollection goldSents = getData(gold, type, "gold");
        AnnoSentenceCollection predSents1 = getData(pred1, type, "pred1");
        AnnoSentenceCollection predSents2 = getData(pred2, type, "pred2");
        
        // 20 seconds for 2416 sentences.
        double ppt = pptDepParseAccuracy(goldSents, predSents1, predSents2, skipPunctuation);
        log.info("p-value (paired permutation): {}", ppt);
        // 126 seconds for 2416 sentences.
        double bts = bootstrapDepParseAccuracy(goldSents, predSents1, predSents2, skipPunctuation);
        log.info("p-value (paired bootstrap): {}", bts);
    }

    protected static AnnoSentenceCollection getData(File path, DatasetType type, String name) throws IOException {
        AnnoSentenceReaderPrm prm = new AnnoSentenceReaderPrm();
        prm.name = name;
        AnnoSentenceReader reader = new AnnoSentenceReader(prm);
        reader.loadSents(path, type);
        return reader.getData();
    }
    
}
