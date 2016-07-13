package edu.jhu.nlp.eval;

import java.io.File;
import java.io.IOException;

import org.apache.commons.cli.ParseException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.EvalPipeline;
import edu.jhu.nlp.data.simple.AnnoSentenceCollection;
import edu.jhu.nlp.data.simple.AnnoSentenceReader;
import edu.jhu.nlp.data.simple.AnnoSentenceReader.AnnoSentenceReaderPrm;
import edu.jhu.nlp.data.simple.AnnoSentenceReader.DatasetType;
import edu.jhu.nlp.data.simple.CorpusHandler;
import edu.jhu.nlp.eval.SrlEvaluator.SrlEvaluatorPrm;
import edu.jhu.nlp.features.TemplateLanguage.AT;
import edu.jhu.pacaya.util.Threads;
import edu.jhu.pacaya.util.cli.ArgParser;
import edu.jhu.pacaya.util.cli.Opt;
import edu.jhu.pacaya.util.report.ReporterManager;
import edu.jhu.prim.util.random.Prng;

public class EvalRunner {

    private static final Logger log = LoggerFactory.getLogger(EvalRunner.class);

    // Options not specific to the model
    @Opt(name = "seed", hasArg = true, description = "Pseudo random number generator seed for everything else.")
    public static long seed = Prng.DEFAULT_SEED;
    @Opt(hasArg = true, description = "Number of threads for computation.")
    public static int threads = 1;
    // Options for evaluation.
    @Opt(hasArg=true, description="Whether to skip punctuation in dependency parse evaluation.")
    public static boolean dpSkipPunctuation = false;
    

    // Options for data
    @Opt(hasArg = true, required = true, description = "Predicted data input file or directory.")
    public static File pred = null;
    @Opt(hasArg = true, required = true, description = "Gold data input file or directory.")
    public static File gold = null;
    @Opt(hasArg = true, required = true, description = "Type of data.")
    public static DatasetType type = null;
    
    public static AnnoSentenceCollection getData(File path, String name, DatasetType type) throws IOException {
        AnnoSentenceReaderPrm prm = new AnnoSentenceReaderPrm();
        prm.name = name;
        AnnoSentenceReader reader = new AnnoSentenceReader(prm);
        reader.loadSents(path, type);
        return reader.getData();
    }
    
    public static void run() throws IOException {
        // The evaluation pipeline.
        EvalPipeline eval = new EvalPipeline();

        // Pruning
        eval.add(new PruningEfficiency(dpSkipPunctuation));
        eval.add(new OraclePruningAccuracy(dpSkipPunctuation));
        eval.add(new OraclePruningExactMatch(dpSkipPunctuation));
        // POS Tagging.
        eval.add(new PosTagAccuracy());
        // Dependency Parsing.
        eval.add(new DepParseAccuracy(dpSkipPunctuation));
        eval.add(new DepParseExactMatch(dpSkipPunctuation));
        // SRL...
        eval.add(new SrlPredIdAccuracy());
        eval.add(new SrlSelfLoops());
        // Unlabled predicate position identification.
        eval.add(new SrlEvaluator(new SrlEvaluatorPrm(false, false, true, false)));
        // Labeled predicate sense classification.
        eval.add(new SrlEvaluator(new SrlEvaluatorPrm(true, true, false, false)));
        // SRL without sense, assuming predicate positions are given.
        eval.add(new SrlEvaluator(new SrlEvaluatorPrm(true, false, false, true)));
        // Full SRL, assuming predicate positions are given.
        eval.add(new SrlEvaluator(new SrlEvaluatorPrm(true, true, false, true)));
        // Relation extraction.
        eval.add(new RelationEvaluator());
        // Proportion of annotations.
        eval.add(new ProportionAnnotated(AT.values()));
        
        // Read data.
        AnnoSentenceCollection predSents = getData(pred, "pred", type);
        AnnoSentenceCollection goldSents = getData(gold, "gold", type);
        
        // Evaluate.
        eval.evaluate(predSents, goldSents, "pred/gold");
    }

    public static void main(String[] args) {
        int exitCode = 0;
        ArgParser parser = null;
        try {
            parser = new ArgParser(EvalRunner.class);
            parser.registerClass(EvalRunner.class);
            parser.registerClass(ReporterManager.class);
            parser.parseArgs(args);
            
            ReporterManager.init(ReporterManager.reportOut, true);
            Prng.seed(seed);
            Threads.initDefaultPool(threads);

            EvalRunner.run();
        } catch (ParseException e1) {
            log.error(e1.getMessage());
            if (parser != null) {
                parser.printUsage();
            }
            exitCode = 1;
        } catch (Throwable t) {
            t.printStackTrace();
            exitCode = 1;
        }
        
        System.exit(exitCode);
    }
    
}
