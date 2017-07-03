package edu.jhu.nlp.data.simple;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.AnnoPipeline;
import edu.jhu.nlp.relations.RelationMunger;
import edu.jhu.nlp.relations.RelationMunger.RelationDataPostproc;
import edu.jhu.nlp.relations.RelationMunger.RelationDataPreproc;
import edu.jhu.nlp.relations.RelationMunger.RelationMungerPrm;
import edu.jhu.pacaya.util.cli.ArgParser;
import edu.jhu.pacaya.util.cli.Opt;

public class BIOTagConverter {
    
    private static final Logger log = LoggerFactory.getLogger(BIOTagConverter.class);

    @Opt(description="Convert CoNLL-2003 BIO tagging to CoNLL-2002 style BIO tagging.")
    public static boolean c03Toc02 = true;
        
    public static void conll03ToConll02(AnnoSentence sent) {
        sent.setNeTags(conll03ToConll02(sent.getNeTags()));
        sent.setChunks(conll03ToConll02(sent.getChunks()));
    }
    
    public static List<String> conll03ToConll02(List<String> origTags) {
        if (origTags == null) {
            return origTags;
        }
        List<String> newTags = new ArrayList<>();
        String prev = "O";
        for (int i=0; i<origTags.size(); i++) {
            String cur = origTags.get(i);
            String newCur = cur;
            if (cur.startsWith("I-") && (prev.equals("O") || !prev.equals(cur))) {
                newCur = "B-" + cur.substring(2);
            }
            prev = cur;
            newTags.add(newCur);
        }
        return newTags;
    }
    
    private static void run(ArgParser parser) throws IOException {
        CorpusHandler handler = new CorpusHandler();
        AnnoSentenceCollection sents = handler.getTrainGold();
        for (AnnoSentence sent : sents) {
            if (c03Toc02) {
                conll03ToConll02(sent);
            }
        }
        handler.writeTrainGold();
    }
    
    public static void main(String[] args) throws IOException {
        ArgParser parser = new ArgParser(BIOTagConverter.class);
        parser.registerClass(BIOTagConverter.class);
        parser.registerClass(CorpusHandler.class);
        parser.parseArgs(args);

        BIOTagConverter.run(parser);
    }
    
}
