package edu.jhu.nlp.features;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
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
import edu.jhu.nlp.data.simple.AlphabetStore;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.data.simple.AnnoSentenceCollection;
import edu.jhu.nlp.data.simple.AnnoSentenceReader.DatasetType;
import edu.jhu.nlp.data.simple.AnnoSentenceReaderSpeedTest;
import edu.jhu.nlp.data.simple.IntAnnoSentence;
import edu.jhu.nlp.features.TemplateLanguage.AT;
import edu.jhu.nlp.features.TemplateLanguage.FeatTemplate;
import edu.jhu.nlp.features.TemplateLanguage.TokProperty;
import edu.jhu.nlp.words.PrefixAnnotator;
import edu.jhu.pacaya.util.hash.MurmurHash;
import edu.jhu.prim.iter.IntIter;
import edu.jhu.prim.list.IntArrayList;
import edu.jhu.prim.set.IntHashSet;
import edu.jhu.prim.util.Timer;
import edu.jhu.prim.util.math.FastMath;

public class IntTemplateFeatureExtractorCollisionsTest {
    
    private static final Logger log = LoggerFactory.getLogger(IntTemplateFeatureExtractorCollisionsTest.class);
    
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
    private static final int featureHashMod = 1000000;
    private static int maxSents = 200;
    private static int trials = 1; 
    private static Path colPath = Paths.get("./collisions.txt");
    private static boolean useIntExtr = true;
    
    // Global counters.
    private static Set<String>[] collisions;
    private static int[] collisionTokenCount;

    // Note: prefer >= 2 trials since the first one will initialize the alphabet.
    private static double run(List<FeatTemplate> tpls, FeatureForm form) { 
        AnnoSentenceCollection sents = AnnoSentenceReaderSpeedTest.read(AnnoSentenceReaderSpeedTest.c09Dev, DatasetType.CONLL_2009);
        PrefixAnnotator.addPrefixes(sents);
        CorpusStatistics cs = new CorpusStatistics(new CorpusStatisticsPrm());
        AlphabetStore store = new AlphabetStore(sents);
        
        maxSents = Math.min(maxSents, sents.size());
        
        double msPerSent = 0;
        double toksPerSec = 0;
        for (int trial = 0; trial < trials; trial++) {
            Timer timer = new Timer();
            timer.start();
            int n=0;
            for (int s=0; s<maxSents; s++) {
                AnnoSentence sent = sents.get(s);
                TemplateFeatureExtractor extStr = new TemplateFeatureExtractor(sent, cs);
                IntTemplateFeatureExtractor extInt = new IntTemplateFeatureExtractor(new IntAnnoSentence(sent, store), cs);
                if (form == FeatureForm.DEP) {                    
                    for (int i = -1; i < sent.size(); i++) {
                        for (int j = 0; j < sent.size(); j++) {
                            LocalObservations local = LocalObservations.newPidxCidx(i, j);
                            getFeatures(tpls, extStr, extInt, local);
                        }
                    }
                } else if (form == FeatureForm.POS) {
                    for (int i = -1; i < sent.size(); i++) {
                        LocalObservations local = LocalObservations.newPidx(i);
                        getFeatures(tpls, extStr, extInt, local);
                    }
                } else if (form == FeatureForm.SRL) {
                    IntHashSet known = sent.getKnownPreds();
                    IntIter iter = known.iterator();
                    while (iter.hasNext()) {
                        int i = iter.next();
                        for (int j = 0; j < sent.size(); j++) {
                            LocalObservations local = LocalObservations.newPidxCidx(i, j);
                            getFeatures(tpls, extStr, extInt, local);
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
        }
        return msPerSent;
    }
    
    protected static void getFeatures(List<FeatTemplate> tpls, TemplateFeatureExtractor extStr, IntTemplateFeatureExtractor extInt,
            LocalObservations local) {        
        for (FeatTemplate tpl : tpls) {
            List<String> featsStr = new ArrayList<>();
            extStr.addFeatures(tpl, local, featsStr);
            IntArrayList featsInt = new IntArrayList();
            extInt.addFeatures(tpl, local, featsInt);
            if (featsStr.size() != featsInt.size()) {
                log.error("Mismatch in number of features extracted for template: {} str = {} int = {}", tpl.getName(), featsStr.size(), featsInt.size());
            }
            for (int i=0; i<Math.min(featsStr.size(), featsInt.size()); i++) {
                int hash;
                if (useIntExtr) {
                    hash = featsInt.get(i);
                } else {
                    hash = MurmurHash.hash32(featsStr.get(i));
                }
                hash = FastMath.mod(hash, featureHashMod);
                // Add to the set of strings which are colliding on this hash.
                Set<String> values = collisions[hash];
                if (values == null) {
                    values = new HashSet<String>();
                    collisions[hash] = values;
                }
                values.add(featsStr.get(i));
                // Increment the token count of collisions on this hash.
                collisionTokenCount[hash] += 1;
            }
        }
    }
    
    public static void initCollisions() {
        collisions = new Set[featureHashMod];
        collisionTokenCount = new int[featureHashMod];
    }
    
    public static void writeCollisionStats(Path outPath) throws IOException {        
        try (BufferedWriter colWriter = Files.newBufferedWriter(outPath, Charset.forName("UTF-8"))) {
            assert collisions.length == collisionTokenCount.length;
            colWriter.write(String.format("%10s %10s %10s %10s", "tok-count", "type-count", "feat-hash", "feat-str"));
            for (int hash=0; hash<collisions.length; hash++) {
                Set<String> values = collisions[hash];
                if (values == null) { continue; }
                colWriter.write(String.format("%10d %10d %10d ", collisionTokenCount[hash], values.size(), hash));
                for (String val : values) {
                    colWriter.write(val);
                    colWriter.write(" ");
                }
                colWriter.write("\n");
            }
        }
    }    
    
    /** Emprical rate of type collisions. */
    public static double getTypeColRate() {
        int typeColCount = 0;
        int typeTotal = 0;
        for (int hash=0; hash<collisions.length; hash++) {
            Set<String> values = collisions[hash];
            if (values == null) { continue; }
            int binSize = values.size();
            typeTotal += binSize;
            if (binSize > 1) {
                typeColCount += binSize;
            }
        }
        return (double) typeColCount / typeTotal;
    }

    /** Emprical rate of token collisions. */
    public static double getTokColRate() {
        int tokColCount = 0;
        int tokTotal = 0;
        for (int hash=0; hash<collisions.length; hash++) {
            Set<String> values = collisions[hash];
            if (values == null) { continue; }
            int binSize = values.size();
            tokTotal += collisionTokenCount[hash];
            if (binSize > 1) {
                tokColCount += collisionTokenCount[hash];
            }
        }
        return (double) tokColCount / tokTotal;
    }

    /** Total number of feature types. */
    public static double getTypeTotal() {
        int typeTotal = 0;
        for (int hash=0; hash<collisions.length; hash++) {
            int binSize = collisions[hash].size();
            typeTotal += binSize;
        }
        return typeTotal;
    }

    /*
     * %-type-col  %-tok-col      Notes
     *   0.290008   0.334824 useIntExtr=true
     *   0.275462   0.283401 useIntExtr=false
     */
    public static void main(String[] args) throws ParseException, IOException {
        maxSents = 400;
        
        useIntExtr = true;
        initCollisions();
        testSpeedSrlC1();
        writeCollisionStats(Paths.get("./collisions-1.txt"));
        double typeColRate1 = getTypeColRate();
        double tokColRate1 = getTokColRate();

        useIntExtr = false;
        initCollisions();
        testSpeedSrlC1();
        writeCollisionStats(Paths.get("./collisions-2.txt"));
        double typeColRate2 = getTypeColRate();
        double tokColRate2 = getTokColRate();
        
        System.out.printf("%10s %10s %10s\n", "%-type-col", "%-tok-col", "Notes");
        System.out.printf("%10f %10f %10s\n", typeColRate1, tokColRate1, "useIntExtr=true");
        System.out.printf("%10f %10f %10s\n", typeColRate2, tokColRate2, "useIntExtr=false");        
    }
    
}
