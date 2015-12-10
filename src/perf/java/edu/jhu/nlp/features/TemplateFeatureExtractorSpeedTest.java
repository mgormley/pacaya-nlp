package edu.jhu.nlp.features;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.cli.ParseException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.CorpusStatistics;
import edu.jhu.nlp.CorpusStatistics.CorpusStatisticsPrm;
import edu.jhu.nlp.data.simple.AlphabetStore;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.data.simple.AnnoSentenceCollection;
import edu.jhu.nlp.data.simple.AnnoSentenceReader.DatasetType;
import edu.jhu.nlp.data.simple.AnnoSentenceReaderSpeedTest;
import edu.jhu.nlp.data.simple.IntAnnoSentence;
import edu.jhu.nlp.features.TemplateLanguage.AT;
import edu.jhu.nlp.features.TemplateLanguage.FeatTemplate;
import edu.jhu.nlp.features.TemplateLanguage.TokProperty;
import edu.jhu.nlp.relations.FeatureUtils;
import edu.jhu.nlp.words.PrefixAnnotator;
import edu.jhu.pacaya.gm.feat.FeatureVector;
import edu.jhu.pacaya.util.FeatureNames;
import edu.jhu.prim.iter.IntIter;
import edu.jhu.prim.list.IntArrayList;
import edu.jhu.prim.set.IntHashSet;
import edu.jhu.prim.util.Timer;

public class TemplateFeatureExtractorSpeedTest {
    
    private static final Logger log = LoggerFactory.getLogger(TemplateFeatureExtractorSpeedTest.class);
    private static final int featureHashMod = 1000000;
    
    /**
     * (OLD) Speed test results.
     * 
     * Gilim:
     *    w/hash s=800 n=19560 Toks / sec: 238.92
     *    w/o    s=800 n=19560 Toks / sec: 560.39
     *    
     * Shasta:
     * 	  w/hash s=800 n=19560 Toks / sec: 375.66 (was 338.00 w/Feature object)
     * 	  w/o    s=800 n=19560 Toks / sec: 723.21
     */
    //@Test
    public static double testSpeedDepParse() throws ParseException, IOException {
        List<FeatTemplate> tpls = TemplateSets.getFromResource(TemplateSets.mcdonaldDepFeatsResource);
        return run(tpls, FeatureForm.DEP);
    }
    
    //@Test
    public static double testSpeedPosTag() throws ParseException, IOException {
        List<FeatTemplate> tpls = TemplateSets.getFromResource(TemplateSets.custom3TagFeatsResource);
        return run(tpls, FeatureForm.POS);
    }

    //@Test
    public static double testSpeedSrlC1() throws ParseException, IOException {
        List<FeatTemplate> tpls = TemplateSets.getFromResource(TemplateSets.coarse1ArgFeatsResource);
        tpls = TemplateLanguage.filterOutRequiring(tpls, AT.BROWN);
        tpls = TemplateLanguage.filterOutFeats(tpls, TokProperty.WORD_TOP_N);
        return run(tpls, FeatureForm.SRL);
    }
    
    //@Test
    public static double testSpeedSrlC1En() throws ParseException, IOException {
        List<FeatTemplate> tpls = TemplateSets.getFromResource(TemplateSets.coarse1EnArgFeatsResource);
        tpls = TemplateLanguage.filterOutRequiring(tpls, AT.BROWN);
        tpls = TemplateLanguage.filterOutFeats(tpls, TokProperty.WORD_TOP_N);
        return run(tpls, FeatureForm.SRL);
    }
    
    public enum FeatureForm { DEP, POS, SRL }
    
    // Global parameters.
    private static int maxSents = 200;
    private static int trials = 1; 
    private static boolean useAlphabet = false;
    private static boolean useStrs = true;
    
    // Note: prefer >= 2 trials since the first one will initialize the alphabet.
    private static double run(List<FeatTemplate> tpls, FeatureForm form) { 
        AnnoSentenceCollection sents = AnnoSentenceReaderSpeedTest.read(AnnoSentenceReaderSpeedTest.c09Dev, DatasetType.CONLL_2009);
        PrefixAnnotator.addPrefixes(sents);
        CorpusStatistics cs = new CorpusStatistics(new CorpusStatisticsPrm());
        AlphabetStore store = new AlphabetStore(sents);
        
        maxSents = Math.min(maxSents, sents.size());
        
        FeatureNames alphabet = new FeatureNames();
        
        double msPerSent = 0;
        double toksPerSec = 0;
        for (int trial = 0; trial < trials; trial++) {
            Timer timer = new Timer();
            timer.start();
            int n=0;
            for (int s=0; s<maxSents; s++) {
                AnnoSentence sent = sents.get(s);
                TemplateFeatureExtractor extStr = null;
                IntTemplateFeatureExtractor extInt = null;
                if (useStrs) {
                    extStr = new TemplateFeatureExtractor(sent, cs);
                } else {
                    extInt = new IntTemplateFeatureExtractor(new IntAnnoSentence(sent, store), cs);
                }
                if (form == FeatureForm.DEP) {                    
                    for (int i = -1; i < sent.size(); i++) {
                        for (int j = 0; j < sent.size(); j++) {
                            LocalObservations local = LocalObservations.newPidxCidx(i, j);
                            FeatureVector fv = getFeatures(tpls, alphabet, extStr, extInt, local);
                        }
                    }
                } else if (form == FeatureForm.POS) {
                    for (int i = -1; i < sent.size(); i++) {
                        LocalObservations local = LocalObservations.newPidx(i);
                        FeatureVector fv = getFeatures(tpls, alphabet, extStr, extInt, local);
                    }
                } else if (form == FeatureForm.SRL) {
                    IntHashSet known = sent.getKnownPreds();
                    IntIter iter = known.iterator();
                    while (iter.hasNext()) {
                        int i = iter.next();
                        for (int j = 0; j < sent.size(); j++) {
                            LocalObservations local = LocalObservations.newPidxCidx(i, j);
                            FeatureVector fv = getFeatures(tpls, alphabet, extStr, extInt, local);
                        }
                    }
                } else {
                    throw new RuntimeException();
                }
                    
                timer.stop();
                n += sent.size();
                if (s % 100 == 0) {
                    log.info("s="+s+" n=" + n + " Toks / sec: " + (n / timer.totSec())); 
                }
                timer.start();
            }
            timer.stop();
            msPerSent = timer.totMs() / maxSents;
            toksPerSec = n / timer.totSec();
            log.info("Average ms per sent: " + msPerSent);
            log.info("Toks / sec: " + toksPerSec);
            if (useAlphabet) { log.info("Alphabet.size(): " + alphabet.size()); }
        }
        return msPerSent;
    }

    protected static FeatureVector getFeatures(List<FeatTemplate> tpls, FeatureNames alphabet, TemplateFeatureExtractor extStr,
            IntTemplateFeatureExtractor extInt, LocalObservations local) {
        if (useStrs) {
            ArrayList<String> feats = new ArrayList<String>();
            extStr.addFeatures(tpls, local, feats );
            FeatureVector fv = new FeatureVector(feats.size());
            if (useAlphabet) {
                FeatureUtils.addFeatures(feats, alphabet, fv, false, featureHashMod);
            } else {
                FeatureUtils.addFeatures(feats, fv, featureHashMod);
            }
            return fv;
        } else {
            IntArrayList feats = new IntArrayList();
            extInt.addFeatures(tpls, local, feats);
            FeatureVector fv = new FeatureVector(feats.size());
            FeatureUtils.addFeatures(feats, fv, featureHashMod);
            return fv;
        }
    }
    
    /*

Speed test results:
            Dep             POS         SRL(C1)       SRL(C1en)           Notes
         280.96        85543.86         1348.08         1437.08             str
        1065.33       232190.48         3475.41         3888.36             int
        
            Dep             POS         SRL(C1)       SRL(C1en)           Notes
         271.96        81266.67         1445.60         1545.97             str
         873.37        78645.16         3321.53         3666.17             int        
         966.69       212000.00         3983.66         4083.75             int (after List<Integer> --> IntArrayList)
         
         w/400 sentences
            Dep             POS         SRL(C1)       SRL(C1en)           Notes
          91.54            0.23           17.50           15.66             str
          21.70            0.11            5.85            6.11             int
            
     */
    public static void main(String[] args) throws ParseException, IOException {
        double depStr, posStr, srlC1Str, srlEnStr;
        double depInt, posInt, srlC1Int, srlEnInt;
        maxSents = 400;
        depStr = testSpeedDepParse();
        posStr = testSpeedPosTag();
        srlC1Str = testSpeedSrlC1();
        srlEnStr = testSpeedSrlC1En();
        useStrs = false;
        depInt = testSpeedDepParse();
        posInt = testSpeedPosTag();
        srlC1Int = testSpeedSrlC1();
        srlEnInt = testSpeedSrlC1En();
        System.out.printf("%15s %15s %15s %15s %15s\n", "Dep", "POS", "SRL(C1)", "SRL(C1en)", "Notes");
        System.out.printf("%15.2f %15.2f %15.2f %15.2f %15s\n", depStr, posStr, srlC1Str, srlEnStr, "str");
        System.out.printf("%15.2f %15.2f %15.2f %15.2f %15s\n", depInt, posInt, srlC1Int, srlEnInt, "int");
    }
    
}
