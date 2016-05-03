package edu.jhu.nlp.ner;

import java.io.IOException;

import org.apache.commons.cli.ParseException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.hlt.optimize.Optimizer;
import edu.jhu.hlt.optimize.function.DifferentiableBatchFunction;
import edu.jhu.hlt.optimize.function.DifferentiableFunction;
import edu.jhu.nlp.AnnoPipeline;
import edu.jhu.nlp.Evaluator;
import edu.jhu.nlp.data.simple.AnnoSentenceCollection;
import edu.jhu.nlp.data.simple.CorpusHandler;
import edu.jhu.nlp.eval.NerEvaluator;
import edu.jhu.nlp.joint.OptimizerFactory;
import edu.jhu.nlp.ner.NerAnnotator.NerAnnotatorPrm;
import edu.jhu.nlp.ner.NerFactorGraphBuilder.NerFactorGraphBuilderPrm;
import edu.jhu.pacaya.gm.decode.MbrDecoder.MbrDecoderPrm;
import edu.jhu.pacaya.gm.inf.BeliefPropagation.BeliefPropagationPrm;
import edu.jhu.pacaya.gm.inf.BeliefPropagation.BpScheduleType;
import edu.jhu.pacaya.gm.inf.BeliefPropagation.BpUpdateOrder;
import edu.jhu.pacaya.gm.inf.FgInferencerFactory;
import edu.jhu.pacaya.gm.train.CrfTrainer.CrfTrainerPrm;
import edu.jhu.pacaya.util.Threads;
import edu.jhu.pacaya.util.cli.ArgParser;
import edu.jhu.pacaya.util.cli.Opt;
import edu.jhu.pacaya.util.report.ReporterManager;
import edu.jhu.pacaya.util.semiring.LogSemiring;
import edu.jhu.prim.tuple.Pair;
import edu.jhu.prim.util.random.Prng;

public class NerRunner {

    private static final Logger log = LoggerFactory.getLogger(NerRunner.class);

    // Options not specific to the model
    @Opt(name = "seed", hasArg = true, description = "Pseudo random number generator seed for everything else.")
    public static long seed = Prng.DEFAULT_SEED;
    @Opt(hasArg = true, description = "Number of threads for computation.")
    public static int threads = 1;
    @Opt(hasArg = true, description = "The value of the mod for use in the feature hashing trick. If <= 0, feature-hashing will be disabled.")
    public static int featureHashMod = 1048576; // 2^20

    public static void run() throws IOException {
        // Construct the annotation pipeline.
        AnnoPipeline anno = new AnnoPipeline();
        anno.add(new NerAnnotator(getNerAnnotatorPrm()));
        
        // Construct the evaluator.
        Evaluator eval = new NerEvaluator();
        
        // Initialize the data reader/writer.
        CorpusHandler corpus = new CorpusHandler();
        {
            // Either of train or dev might be null.
            AnnoSentenceCollection trainGold = corpus.getTrainGold();
            AnnoSentenceCollection trainInput = corpus.getTrainInput();
            AnnoSentenceCollection devGold = corpus.getDevGold();
            AnnoSentenceCollection devInput = corpus.getDevInput();

            if (corpus.hasTrain()) {
                // Preprocess the gold train data and write it out.
                corpus.writeTrainGold();
            }
            if (corpus.hasDev()) {
                // Preprocess the gold dev data and write it out.
                corpus.writeDevGold();
            }
            
            if (corpus.hasTrain()) {
                // Train a model. (AnnoPipeline also annotates all the train and dev input.)
                corpus.writeTrainGold();
                anno.train(trainInput, trainGold, devInput, devGold);                
            } else if (corpus.hasDev()) { // but not train
                corpus.writeDevGold();
                anno.annotate(devInput);
            }
            
            if (corpus.hasTrain()) {
                // Decode and evaluate. the train data.
                corpus.writeTrainPreds(trainInput);
                eval.evaluate(trainInput, trainGold, "train");
                corpus.clearTrainCache();
            }
            if (corpus.hasDev()) {
                // Decode and evaluate the dev data.
                corpus.writeDevPreds(devInput);
                eval.evaluate(devInput, devGold, "dev");
                corpus.clearDevCache();
            }
        }
        
        if (corpus.hasTest()) {
            // Decode test data.
            String name = "test";
            AnnoSentenceCollection testInput = corpus.getTestInput();
            anno.annotate(testInput);
            corpus.writeTestPreds(testInput);
            // Evaluate test data.
            AnnoSentenceCollection testGold = corpus.getTestGold();
            corpus.writeTestGold();
            eval.evaluate(testInput, testGold, name);
            corpus.clearTestCache();
        }
    }

    protected static NerAnnotatorPrm getNerAnnotatorPrm() {
        NerAnnotatorPrm prm = new NerAnnotatorPrm();
        prm.crfPrm = getCrfTrainerPrm();
        prm.nerPrm = getNerFactorGraphBuilderPrm();
        prm.mbrPrm = getMbrDecoderPrm();
        return prm;
    }
    
    private static NerFactorGraphBuilderPrm getNerFactorGraphBuilderPrm() {
        NerFactorGraphBuilderPrm prm = new NerFactorGraphBuilderPrm();
        prm.featureHashMod = featureHashMod;
        return prm;
    }

    private static CrfTrainerPrm getCrfTrainerPrm() {
        CrfTrainerPrm crfPrm = new CrfTrainerPrm();
        Pair<Optimizer<DifferentiableFunction>, Optimizer<DifferentiableBatchFunction>> pair = OptimizerFactory.getOptimizers();
        crfPrm.optimizer = pair.get1();
        crfPrm.batchOptimizer = pair.get2();
        crfPrm.infFactory = getFgInferencerFactory();
        return crfPrm;
    }
    
    private static MbrDecoderPrm getMbrDecoderPrm() {
        MbrDecoderPrm prm = new MbrDecoderPrm();
        prm.infFactory = getFgInferencerFactory();
        return prm;
    }

    private static FgInferencerFactory getFgInferencerFactory() {
        BeliefPropagationPrm prm = new BeliefPropagationPrm();
        prm.keepTape = false;
        prm.schedule = BpScheduleType.TREE_LIKE;
        prm.updateOrder = BpUpdateOrder.SEQUENTIAL;
        prm.maxIterations = 1;
        prm.s = LogSemiring.getInstance();
        return prm;
    }

    public static void main(String[] args) {
        int exitCode = 0;
        ArgParser parser = null;
        try {
            parser = new ArgParser(NerRunner.class);
            parser.registerClass(NerRunner.class);
            parser.registerClass(OptimizerFactory.class);
            parser.registerClass(CorpusHandler.class);
            parser.registerClass(ReporterManager.class);
            parser.parseArgs(args);
            
            ReporterManager.init(ReporterManager.reportOut, true);
            Prng.seed(seed);
            Threads.initDefaultPool(threads);

            NerRunner.run();
        } catch (ParseException e1) {
            log.error(e1.getMessage());
            if (parser != null) {
                parser.printUsage();
            }
            exitCode = 1;
        } catch (Throwable t) {
            t.printStackTrace();
            exitCode = 1;
        } finally {
            Threads.shutdownDefaultPool();
            ReporterManager.close();
        }
        
        System.exit(exitCode);
    }
    
}
