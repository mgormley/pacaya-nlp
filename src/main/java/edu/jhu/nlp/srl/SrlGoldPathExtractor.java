package edu.jhu.nlp.srl;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.commons.cli.ParseException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.CorpusStatistics;
import edu.jhu.nlp.CorpusStatistics.CorpusStatisticsPrm;
import edu.jhu.nlp.data.DepGraph;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.data.simple.AnnoSentenceCollection;
import edu.jhu.nlp.data.simple.AnnoSentenceReader;
import edu.jhu.nlp.data.simple.AnnoSentenceReader.AnnoSentenceReaderPrm;
import edu.jhu.nlp.data.simple.AnnoSentenceReader.DatasetType;
import edu.jhu.nlp.data.simple.CorpusHandler;
import edu.jhu.nlp.features.LocalObservations;
import edu.jhu.nlp.features.TemplateFeatureExtractor;
import edu.jhu.nlp.features.TemplateLanguage.EdgeProperty;
import edu.jhu.nlp.features.TemplateLanguage.FeatTemplate;
import edu.jhu.nlp.features.TemplateLanguage.FeatTemplate3;
import edu.jhu.nlp.features.TemplateLanguage.JoinTemplate;
import edu.jhu.nlp.features.TemplateLanguage.ListModifier;
import edu.jhu.nlp.features.TemplateLanguage.PositionList;
import edu.jhu.pacaya.util.Threads;
import edu.jhu.pacaya.util.cli.ArgParser;
import edu.jhu.pacaya.util.cli.Opt;
import edu.jhu.pacaya.util.report.ReporterManager;
import edu.jhu.prim.util.random.Prng;

public class SrlGoldPathExtractor {

    private static final Logger log = LoggerFactory.getLogger(SrlGoldPathExtractor.class);

    // Options not specific to the model
    @Opt(name = "seed", hasArg = true, description = "Pseudo random number generator seed for everything else.")
    public static long seed = Prng.DEFAULT_SEED;
    @Opt(hasArg = true, description = "Number of threads for computation.")
    public static int threads = 1;
    // Options for evaluation.
    @Opt(hasArg=true, description="Whether to skip punctuation in dependency parse evaluation.")
    public static boolean dpSkipPunctuation = false;
    
    // Options for data
    @Opt(hasArg = true, description = "Data input file or directory.")
    public static File data = null;
    @Opt(hasArg = true, description = "Type of data.")
    public static DatasetType type = null;
    
    // Feature templates for extracting the paths
    private static FeatTemplate3 tpl1 = new FeatTemplate3(PositionList.PATH_P_C, null, EdgeProperty.DIR, ListModifier.SEQ);
    private static FeatTemplate3 tpl2 = new FeatTemplate3(PositionList.PATH_P_C, null, EdgeProperty.EDGEREL, ListModifier.SEQ);
    private static JoinTemplate tpl = new JoinTemplate(tpl1, tpl2);

    public static void run() throws IOException {        
        // Read data.
        AnnoSentenceCollection sents = getData(data, "input", type);
        
        CorpusStatistics cs = new CorpusStatistics(new CorpusStatisticsPrm());
        cs.init(sents);

        Set<String> paths = getGoldSrlPaths(sents, cs);
        
        log.info("Number of paths: " + paths.size());
        BufferedWriter writer = Files.newBufferedWriter(Paths.get("paths.txt"));
        for (String path : paths) {
            writer.write(path);
            writer.write("\n");
        }
        writer.close();
    }
    
    public static Set<String> getGoldSrlPaths(AnnoSentenceCollection sents, CorpusStatistics cs) {        
        Set<String> paths = new HashSet<>();
        for (AnnoSentence sent : sents) {
            DepGraph srl = sent.getSrlGraph();
            for (int i=0; i<sent.size(); i++) {
                for (int j=0; j<sent.size(); j++) {
                    if (srl.get(i, j) != null) {
                        List<String> feats = new ArrayList<>();
                        TemplateFeatureExtractor ext = new TemplateFeatureExtractor(sent, cs);
                        ext.addFeatures(tpl, LocalObservations.newPidxCidx(i, j), feats);
                        paths.addAll(feats);
                    }
                }
            }
        }
        return paths;
    }
    
    public static AnnoSentenceCollection getData(File path, String name, DatasetType type) throws IOException {
        AnnoSentenceReaderPrm prm = new AnnoSentenceReaderPrm();
        prm.name = name;
        AnnoSentenceReader reader = new AnnoSentenceReader(prm);
        reader.loadSents(path, type);
        return reader.getData();
    }
    
    public static void main(String[] args) {
        int exitCode = 0;
        ArgParser parser = null;
        try {
            parser = new ArgParser(SrlGoldPathExtractor.class);
            parser.registerClass(SrlGoldPathExtractor.class);
            parser.registerClass(CorpusHandler.class);
            parser.registerClass(ReporterManager.class);
            parser.parseArgs(args);
            
            ReporterManager.init(ReporterManager.reportOut, true);
            Prng.seed(seed);
            Threads.initDefaultPool(threads);

            SrlGoldPathExtractor.run();
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
