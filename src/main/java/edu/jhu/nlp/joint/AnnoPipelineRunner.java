package edu.jhu.nlp.joint;

import java.io.File;
import java.io.IOException;

import org.apache.commons.cli.ParseException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.AnnoPipeline;
import edu.jhu.nlp.data.simple.AnnoSentenceCollection;
import edu.jhu.nlp.data.simple.CorpusHandler;
import edu.jhu.nlp.eval.ProportionAnnotated;
import edu.jhu.pacaya.util.Threads;
import edu.jhu.pacaya.util.cli.ArgParser;
import edu.jhu.pacaya.util.cli.Opt;
import edu.jhu.pacaya.util.files.QFiles;
import edu.jhu.pacaya.util.report.Reporter;
import edu.jhu.pacaya.util.report.ReporterManager;
import edu.jhu.prim.util.Timer;
import edu.jhu.prim.util.random.Prng;

/**
 * Pipeline runner for SRL experiments.
 * @author mgormley
 * @author mmitchell
 */
public class AnnoPipelineRunner {

    private static final Logger log = LoggerFactory.getLogger(AnnoPipelineRunner.class);
    private static final Reporter rep = Reporter.getReporter(AnnoPipelineRunner.class);

    // Options not specific to the model
    @Opt(name = "seed", hasArg = true, description = "Pseudo random number generator seed for everything else.")
    public static long seed = Prng.DEFAULT_SEED;
    @Opt(hasArg = true, description = "Number of threads for computation.")
    public static int threads = 1;
    @Opt(hasArg = true, description = "Whether to use a log-add table for faster computation.")
    public static boolean useLogAddTable = false;
    
    // Options for model IO
    @Opt(hasArg = true, required = true, description = "File from which to read a serialized pipeline.")
    public static File pipeIn = null;
        
    public AnnoPipelineRunner() {
    }

    public void run() throws IOException {  
        Timer t = new Timer();
        t.start();
        // Initialize the data reader/writer.
        CorpusHandler corpus = new CorpusHandler();

        AnnoPipeline anno = (AnnoPipeline) QFiles.deserialize(pipeIn);
        if (corpus.hasTest()) {
            // Decode test data.
            AnnoSentenceCollection testInput = corpus.getTestInput();
            anno.annotate(testInput);
            (new ProportionAnnotated(CorpusHandler.getPredAts())).evaluate(testInput, null, "test");
            corpus.writeTestPreds(testInput);
            corpus.clearTestCache();
        }
        t.stop();
        rep.report("elapsedSec", t.totSec());
    }

    public static void main(String[] args) throws IOException {
        ArgParser parser = new ArgParser(AnnoPipelineRunner.class);
        parser.registerClass(AnnoPipelineRunner.class);
        parser.registerClass(CorpusHandler.class);
        parser.registerClass(ReporterManager.class);
        parser.parseArgs(args);
        
        ReporterManager.init(ReporterManager.reportOut, true);
        Prng.seed(seed);
        Threads.initDefaultPool(threads);

        AnnoPipelineRunner pipeline = new AnnoPipelineRunner();
        pipeline.run();
    }

}
