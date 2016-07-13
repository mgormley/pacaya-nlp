package edu.jhu.nlp.eval;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.apache.commons.cli.ParseException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.data.simple.AnnoSentenceCollection;
import edu.jhu.nlp.data.simple.AnnoSentenceReader;
import edu.jhu.nlp.data.simple.AnnoSentenceReader.AnnoSentenceReaderPrm;
import edu.jhu.nlp.data.simple.AnnoSentenceReader.DatasetType;
import edu.jhu.nlp.eval.SrlEvaluator.SrlEvaluatorPrm;
import edu.jhu.nlp.relations.RelationMunger;
import edu.jhu.nlp.relations.RelationMunger.RelationMungerPrm;
import edu.jhu.pacaya.util.cli.ArgParser;
import edu.jhu.pacaya.util.cli.Opt;
import edu.jhu.pacaya.util.report.Reporter;
import edu.jhu.pacaya.util.report.ReporterManager;
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
    private static Reporter rep = Reporter.getReporter(SignificanceTests.class);
    
    public interface EvalMetric<X> {
        
        /**
         * Gets the sufficient statistics for the metric.
         * 
         * @param gold The gold sentence.
         * @param pred The predicted sentence.
         * @return The sufficient statistics.
         */
        double[] getSufficientStats(X gold, X pred);

        /**
         * Gets the value of the metric from the sufficient statistics created by getSufficientStats().
         * 
         * @param ss The sufficient statistics.
         * @return The value of the metric.
         */
        double getMetric(double[] sufficientStats);
        
    }
    
    public static class DepAccuracyMetric implements EvalMetric<AnnoSentence> {

        private boolean skipPunctuation;

        public DepAccuracyMetric(boolean skipPunctuation) {
            this.skipPunctuation = skipPunctuation;
        }

        @Override
        public double[] getSufficientStats(AnnoSentence gold, AnnoSentence pred) {
            DepParseAccuracy acc = new DepParseAccuracy(skipPunctuation);
            acc.loss(pred, gold);
            double[] ss = new double[2];
            ss[0] = acc.getCorrect();
            ss[1] = acc.getTotal();
            return ss;
        }

        @Override
        public double getMetric(double[] ss) {
            return (ss[1] != 0) ? ss[0] / ss[1] : 0; 
        }
        
    }

    public static class PosTagAccuracyMetric implements EvalMetric<AnnoSentence> {

        @Override
        public double[] getSufficientStats(AnnoSentence gold, AnnoSentence pred) {
            PosTagAccuracy acc = new PosTagAccuracy();
            acc.loss(pred, gold);
            double[] ss = new double[2];
            ss[0] = acc.getCorrect();
            ss[1] = acc.getTotal();
            return ss;
        }

        @Override
        public double getMetric(double[] ss) {
            return (ss[1] != 0) ? ss[0] / ss[1] : 0; 
        }
        
    }

    public static class SrlF1Metric implements EvalMetric<AnnoSentence> {

        private SrlEvaluatorPrm prm;
        private Metric m;

        public SrlF1Metric(SrlEvaluatorPrm prm, Metric m) {
            assert m.name().startsWith("SRL_");
            this.prm = prm;
            this.m = m;
        }

        @Override
        public double[] getSufficientStats(AnnoSentence gold, AnnoSentence pred) {
            SrlEvaluator eval = new SrlEvaluator(prm);
            eval.accum(gold, pred);
            double[] ss = new double[3];
            ss[0] = eval.getNumCorrectPositive();
            ss[1] = eval.getNumPredictPositive();
            ss[2] = eval.getNumTruePositive();
            return ss;
        }

        @Override
        public double getMetric(double[] ss) {
            double numCorrectPositive = ss[0];
            double numPredictPositive = ss[1];
            double numTruePositive = ss[2];
            double precision = (numPredictPositive == 0) ? 0.0 : numCorrectPositive / numPredictPositive;
            double recall = (numTruePositive == 0) ? 0.0 :  numCorrectPositive / numTruePositive;
            double f1 = (precision == 0.0 && recall == 0.0) ? 0.0 : (2 * precision * recall) / (precision + recall);
            if (m == Metric.SRL_F1) {
                return f1;
            } else if (m == Metric.SRL_P) {
                return precision;
            } else if (m == Metric.SRL_R) {
                return recall;
            } else {
                throw new IllegalArgumentException("Metric must be one of SRL_F1, SRL_P, SRL_R.");
            }
        }
        
    }

    public static class RelF1Metric implements EvalMetric<AnnoSentence> {
        
        private Metric m;

        public RelF1Metric(Metric m) {
            assert m.name().startsWith("REL_");
            this.m = m;
        }
        
        @Override
        public double[] getSufficientStats(AnnoSentence gold, AnnoSentence pred) {
            RelationEvaluator eval = new RelationEvaluator();
            eval.accum(gold.getRelLabels(), pred.getRelLabels());
            double[] ss = new double[3];
            ss[0] = eval.getNumCorrectPositive();
            ss[1] = eval.getNumPredictPositive();
            ss[2] = eval.getNumTruePositive();
            return ss;
        }

        @Override
        public double getMetric(double[] ss) {
            double numCorrectPositive = ss[0];
            double numPredictPositive = ss[1];
            double numTruePositive = ss[2];
            double precision = (numPredictPositive == 0) ? 0.0 : numCorrectPositive / numPredictPositive;
            double recall = (numTruePositive == 0) ? 0.0 :  numCorrectPositive / numTruePositive;
            double f1 = (precision == 0.0 && recall == 0.0) ? 0.0 : (2 * precision * recall) / (precision + recall);  
            if (m == Metric.REL_F1) {
                return f1;
            } else if (m == Metric.REL_P) {
                return precision;
            } else if (m == Metric.REL_R) {
                return recall;
            } else {
                throw new IllegalArgumentException("Metric must be one of REL_F1, REL_P, REL_R.");
            }
        }
        
    }
    
    public static double fastPairedPermutationTestDpAcc(
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
        if (mean == 0) {
            throw new IllegalStateException("no difference between pred1 and pred2");
        }
        // If pred2 scored higher than pred1 we flip all the signs.
        boolean timesNeg1 = (mean < 0);
        if (timesNeg1) { mean *= -1; }
        
        // Shuffle and compute the mean.
        int numGte = 0;
        for (int s=0; s<numSamples; s++) {
            double shufMean = getShuffledMean(diffs);
            if (timesNeg1) { shufMean *= -1; }
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
    
    /** Paired permutation test as given by Yeh (2000). */
    public static <X> double pairedPermutationTest(
            List<X> goldSents, 
            List<X> predSents1, 
            List<X> predSents2, 
            final int numSamples, EvalMetric<X> metric) {
        // Compute the sufficient statistics for each data set.
        double[][] ss1 = new double[goldSents.size()][2];
        double[][] ss2 = new double[goldSents.size()][2];
        for (int i=0; i<goldSents.size(); i++) {
            X gold = goldSents.get(i);
            X pred1 = predSents1.get(i);
            X pred2 = predSents2.get(i);
            ss1[i] = metric.getSufficientStats(gold, pred1);
            ss2[i] = metric.getSufficientStats(gold, pred2);
        }
        
        assert ss1.length == ss2.length;
        assert ss1[0].length == ss2[0].length;
        
        int numSents = ss1.length;
        int numStats = ss1[0].length;

        boolean[] flips = new boolean[numSents];
        {
            // Log the scores for reference.
            double[][] ssEmpty = new double[goldSents.size()][numStats];
            double score1 = getShuffledDiff(ss1, ssEmpty, flips, metric); 
            log.info("Score on dataset 1: " + score1);
            rep.report("score1", score1);
            double score2 = getShuffledDiff(ss2, ssEmpty, flips, metric);
            log.info("Score on dataset 2: " + score2);
            rep.report("score2", score2);
        }
        double diff = getShuffledDiff(ss1, ss2, flips, metric);
        log.trace("diff: " + diff);
        if (diff == 0) {
            throw new IllegalStateException("no difference between pred1 and pred2");
        }
        // If pred2 scored higher than pred1 we flip all the signs.
        boolean timesNeg1 = (diff < 0);
        if (timesNeg1) { diff *= -1; }
        
        // Shuffle and compute the mean.
        int numGte = 0;
        RandBits rand = new RandBits();
        for (int s=0; s<numSamples; s++) {
            for (int i=0; i<numSents; i++) {
                flips[i] = rand.nextBit();
            }
            double diff_s = getShuffledDiff(ss1, ss2, flips, metric);
            if (timesNeg1) { diff_s *= -1; }
            if (diff_s >= diff) {
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

    private static <X> double getShuffledDiff(double[][] ss1, double[][] ss2, boolean[] flips, EvalMetric<X> metric) {
        final int numStats = ss1[0].length;
        // Add the sufficient statistics for the shuffled sample.
        double[] accum1 = new double[numStats];
        double[] accum2 = new double[numStats];
        for (int i=0; i<flips.length; i++) {
            if (flips[i]) {
                DoubleArrays.add(accum1, ss2[i]);
                DoubleArrays.add(accum2, ss1[i]);
            } else {
                DoubleArrays.add(accum1, ss1[i]);
                DoubleArrays.add(accum2, ss2[i]);
            }
        }
        // Compute the metric (accuracy) for each sampled dataset.
        double acc1 = metric.getMetric(accum1);
        double acc2 = metric.getMetric(accum2);
        log.trace("acc1 = {} acc2 = {}", acc1, acc2);
        // Compute the difference in metric.
        double diff = acc1 - acc2;
        return diff;
    }

    /** Paired bootstrap test as given by Berg-Kirkpatrick & Klein (2012). */ 
    public static <X> double bootstrapTest(
            List<X> goldSents, 
            List<X> predSents1, 
            List<X> predSents2, 
            final int numSamples, EvalMetric<X> metric) {
        // Compute the sufficient statistics  for each data set.
        double[][] ss1 = new double[goldSents.size()][2];
        double[][] ss2 = new double[goldSents.size()][2];
        for (int i=0; i<goldSents.size(); i++) {
            X gold = goldSents.get(i);
            X pred1 = predSents1.get(i);
            X pred2 = predSents2.get(i);
            ss1[i] = metric.getSufficientStats(gold, pred1);
            ss2[i] = metric.getSufficientStats(gold, pred2);
        }
        
        assert ss1.length == ss2.length;
        assert ss1[0].length == ss2[0].length;
        
        int numSents = ss1.length;
        int numStats = ss1[0].length;
        
        int numGt = 0;
        // Compute the difference in the metric on the true test set.
        int[] sample = IntArrays.range(numSents);
        {
            // Log the scores for reference.
            double[][] ssEmpty = new double[goldSents.size()][numStats];
            double score1 = computeSampleDiff(ss1, ssEmpty, sample, metric); 
            log.info("Score on dataset 1: " + score1);
            double score2 = computeSampleDiff(ss2, ssEmpty, sample, metric);
            log.info("Score on dataset 2: " + score2);
        }
        double diff = computeSampleDiff(ss1, ss2, sample, metric); 
        log.debug("true diff = {}", diff);
        if (diff == 0) {
            throw new IllegalStateException("no difference between pred1 and pred2");
        }
        // If pred2 scored higher than pred1 we flip all the signs.
        boolean timesNeg1 = (diff < 0);
        if (timesNeg1) { diff *= -1; }
        
        for (int s=0; s<numSamples; s++) {
            // Sample n sentences with replacement.
            for (int ii=0; ii<sample.length; ii++) {
                sample[ii] = Prng.nextInt(numSents);
            }
            // Compute the difference in the metric on the sample.
            double diff_s = computeSampleDiff(ss1, ss2, sample, metric);
            if (timesNeg1) { diff_s *= -1; }
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

    private static <X> double computeSampleDiff(double[][] ss1, double[][] ss2, int[] sample, EvalMetric<X> metric) {
        final int numStats = ss1[0].length;
        // Add the sufficient statistics for the sample.
        double[] accum1 = new double[numStats];
        double[] accum2 = new double[numStats];
        for (int ii=0; ii<sample.length; ii++) {
            int i = sample[ii];
            DoubleArrays.add(accum1, ss1[i]);
            DoubleArrays.add(accum2, ss2[i]);
        }
        // Compute the metric (accuracy) for each sampled dataset.
        double acc1 = metric.getMetric(accum1);
        double acc2 = metric.getMetric(accum2);
        log.trace("acc1 = {} acc2 = {}", acc1, acc2);
        // Compute the difference in metric.
        double diff = acc1 - acc2;
        return diff;
    }
    
    public enum Metric { POS_ACC, DP_ACC, SRL_P, SRL_R, SRL_F1, REL_P, REL_R, REL_F1 } 
    
    @Opt(name="gold", required=true, hasArg=true, description="The path to the gold data")
    public static File _gold = null;
    @Opt(name="pred1", required=true, hasArg=true, description="The path to the predicted data (set 1)")
    public static File _pred1 = null;
    @Opt(name="pred2", required=true, hasArg=true, description="The path to the predicted data (set 2)")
    public static File _pred2 = null;
    @Opt(name="type", required=true, hasArg=true, description="The dataset type")
    public static DatasetType _type = null;
    @Opt(name="metric", required=true, hasArg=true, description="The evaluation metric")
    public static Metric _metric = null;
    
    @Opt(name="skipPunct", required=true, hasArg=true, description="Whether to skip punctuation (dependency accuracy only)")
    public static boolean _skipPunct = false;
    @Opt(name="numSamples", hasArg=true, description="The number of samples to use for the significance test")
    public static int _numSamples = (int) Math.pow(2, 20);
    @Opt(name="maxNumSentences", hasArg=true, description="The maximum number of sentences")
    public static int _maxNumSentences = Integer.MAX_VALUE;

    // TODO: Move this elsewhere?
    public static AnnoSentenceCollection getData(File path, DatasetType type, String name, int maxNumSents) throws IOException {
        AnnoSentenceReaderPrm prm = new AnnoSentenceReaderPrm();
        prm.name = name;
        prm.maxNumSentences = maxNumSents;
        AnnoSentenceReader reader = new AnnoSentenceReader(prm);
        reader.loadSents(path, type);
        AnnoSentenceCollection sents = reader.getData();
        if (type == DatasetType.SEMEVAL_2010) {
            // Munge the relation data into labels.
            RelationMungerPrm rmPrm = new RelationMungerPrm();
            rmPrm.makeRelSingletons = false;
            rmPrm.maxInterveningEntities = -1;
            rmPrm.nePairsFromNeg = false;
            rmPrm.nePairsFromPos = true;
            rmPrm.predictArgRoles = true;
            rmPrm.removeEntityTypes = false;
            rmPrm.shortenEntityMentions = false;
            rmPrm.useRelationSubtype = false;
            RelationMunger rm = new RelationMunger(rmPrm);
            rm.getDataPreproc().annotate(sents);
        }
        return sents;
    }
    
    /**
     * Speed Notes:
     * For unlabeled attachment score on syntactic dependency parses, the
     * speeds were the following for the 2416 sentences of the PTB test set.
     *   - fast paired permutation: 20 seconds
     *   - paired permutation:      73 seconds
     *   - bootstrap:               126 seconds
     */
    public static void main(String[] args) throws ParseException, IOException {
        ArgParser parser = new ArgParser(SignificanceTests.class);
        parser.registerClass(SignificanceTests.class);
        parser.registerClass(ReporterManager.class);
        parser.parseArgs(args);        
        ReporterManager.init(ReporterManager.reportOut, true);

        AnnoSentenceCollection goldSents = getData(_gold, _type, "gold", _maxNumSentences);
        AnnoSentenceCollection predSents1 = getData(_pred1, _type, "pred1", _maxNumSentences);
        AnnoSentenceCollection predSents2 = getData(_pred2, _type, "pred2", _maxNumSentences);
        
        EvalMetric<AnnoSentence> metric;
        if (_metric == Metric.POS_ACC) {
            metric = new PosTagAccuracyMetric(); 
        } else if (_metric == Metric.DP_ACC) {
            metric = new DepAccuracyMetric(_skipPunct);
        } else if (_metric.name().startsWith("SRL_")) {
            SrlEvaluatorPrm prm = new SrlEvaluatorPrm();
            // TODO: add command line options.
            prm.evalPredPosition = false;
            prm.evalRoles = true;
            prm.evalPredSense = true;
            prm.labeled = true;
            metric = new SrlF1Metric(prm, _metric);
        } else if (_metric.name().startsWith("REL_")) {
            metric = new RelF1Metric(_metric);
        } else {
            throw new ParseException("Unsupported metric: " + _metric.name());
        }
        
        if (_metric == Metric.DP_ACC) {
            // 20 seconds for 2416 sentences.
            double ppt = fastPairedPermutationTestDpAcc(goldSents, predSents1, predSents2, _skipPunct);
            log.info("p-value (fast paired permutation): {}", ppt);
            rep.report("p-value-ppt-fast", ppt);
        } else if (_metric.name().startsWith("REL")) {
            RelationEvaluator eval = new RelationEvaluator();
            eval.evaluate(predSents1, goldSents, "pred1");
            eval.evaluate(predSents2, goldSents, "pred2");
        }
        
        double ppt = pairedPermutationTest(goldSents, predSents1, predSents2, _numSamples, metric);
        log.info("p-value (paired permutation): {}", ppt);
        rep.report("p-value-ppt", ppt);
        double bts = bootstrapTest(goldSents, predSents1, predSents2, _numSamples, metric);
        log.info("p-value (paired permutation): {}", bts);
        rep.report("p-value-bts", bts);
    }
    
}
