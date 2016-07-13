package edu.jhu.nlp.data.conll;

import java.io.File;
import java.io.IOException;

import org.apache.commons.cli.ParseException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.tag.BrownClusterTagger;
import edu.jhu.nlp.tag.BrownClusterTagger.BrownClusterTaggerPrm;
import edu.jhu.pacaya.util.cli.ArgParser;
import edu.jhu.pacaya.util.cli.Opt;

public class CoNLL09BrownTagger {

    private static final Logger log = LoggerFactory.getLogger(CoNLL09BrownTagger.class);

    @Opt(name = "train", hasArg = true, required = true, description = "CoNLL 09 input file or directory.")
    public static String train = null;
    @Opt(name = "trainOut", hasArg = true, required = true, description = "CoNLL 09 output output file.")
    public static File trainOut = null;
    @Opt(name = "brownClusters", hasArg = true, required = true, description = "Brown clusters file.")
    public static File brownClusters = null;
    @Opt(name = "maxTagLength", hasArg = true, description = "Maximum length for brown cluster tag.")
    public static int maxTagLength = Integer.MAX_VALUE;

    public static void main(String[] args) throws IOException {
        ArgParser parser = new ArgParser(CoNLL09BrownTagger.class);
        parser.registerClass(CoNLL09BrownTagger.class);
        parser.parseArgs(args);

        BrownClusterTaggerPrm prm = new BrownClusterTaggerPrm();
        prm.maxTagLength = maxTagLength;
        BrownClusterTagger tagger = new BrownClusterTagger(prm);
        tagger.read(brownClusters);
        log.info("Tagging CoNLL data...");
        CoNLL09Reader reader = new CoNLL09Reader(new File(train));
        CoNLL09Writer writer = new CoNLL09Writer(trainOut);
        for (CoNLL09Sentence sent : reader) {
            for (CoNLL09Token tok : sent) {
                String word = tok.getForm();
                // To Lower case...
                word = word.toLowerCase();
                String cluster = tagger.getCluster(word);
                tok.setPos(cluster);
                tok.setPpos(cluster);
            }
            writer.write(sent);
        }
        reader.close();
        writer.close();
        log.info("Done.");
    }
}
