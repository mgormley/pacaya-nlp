package edu.jhu.nlp.eval;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.Evaluator;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.pacaya.util.report.Reporter;
import edu.jhu.prim.tuple.Pair;

/**
 * Computes the precision, recall, and micro-averaged F1 for named entity recognition.
 * 
 * @author mgormley
 */
public class NerEvaluator extends F1Evaluator implements Evaluator {

    private static final Logger log = LoggerFactory.getLogger(NerEvaluator.class);
    private static final Reporter rep = Reporter.getReporter(NerEvaluator.class);

    public NerEvaluator() { }
    
    @Override
    protected Pair<List<String>,List<String>> getLabels(AnnoSentence goldSent, AnnoSentence predSent) {
        List<String> gLabels = new ArrayList<>();
        List<String> pLabels = new ArrayList<>();

        if (goldSent.getNeTags() == null) { return new Pair<>(null, null); }
        if (predSent.getNeTags() == null) { pLabels = null; }
        
        String[][] gChunks = bioTagsToChunks(goldSent.getNeTags());
        String[][] pChunks = bioTagsToChunks(predSent.getNeTags());
        for (int i=0; i<gChunks.length; i++) {
            for (int j=i+1; j<gChunks[i].length; j++) {
                gLabels.add(gChunks[i][j]);
                if (pLabels != null) {
                    pLabels.add(pChunks[i][j]);
                }
            }
        }
        return new Pair<>(gLabels, pLabels);
    }

    private static Pattern tagSplit = Pattern.compile("^([^-]*)-(.*)$");
    
    // In-theory, this supports the same tagging schemes as the conlleval script.
    public static String[][] bioTagsToChunks(List<String> tags) {
        if (tags == null) { return null; }
        // Convert tags to chunks. 
        int n = tags.size();
        String[][] chunks = new String[n+1][n+1];
        int i=-1;
        int j=-1;
        
        String prevTag = "O";
        String prevType = "";
        for (int t=0; t<=n; t++) {
            String tag, type;
            if (t == n) {
                // End-of-sentence token.
                tag = "O";
                type = "";
            } else {
                Matcher m = tagSplit.matcher(tags.get(t));
                if (m.matches()) {
                    tag = m.group(1);
                    type = m.group(2);
                } else {
                    tag = tags.get(t);
                    type = "";
                }
            }
            if (endOfChunk(prevTag, tag, prevType, type)) {
                j = t; // exclusive
                chunks[i][j] = prevType;
                if ("".equals(prevType)) {
                    log.warn("Type missing from previous tag: " + prevTag + "-" + prevType);
                }
            }
            if (startOfChunk(prevTag, tag, prevType, type)) {
                i = t; // inclusive
            }
            prevTag = tag;
            prevType = type;
        }
        return chunks;
    }

    /** Returns true iff a chunk started between the current and previous token. */
    private static boolean startOfChunk(String prevTag, String tag, String prevType, String type) {
        if ("B".equals(prevTag) && "B".equals(tag)) { return true; }
        if ("I".equals(prevTag) && "B".equals(tag)) { return true; }
        if ("O".equals(prevTag) && "B".equals(tag)) { return true; }
        if ("O".equals(prevTag) && "I".equals(tag)) { return true; }
        
        if ("E".equals(prevTag) && "E".equals(tag)) { return true; }
        if ("E".equals(prevTag) && "I".equals(tag)) { return true; }
        if ("O".equals(prevTag) && "E".equals(tag)) { return true; }
        if ("O".equals(prevTag) && "I".equals(tag)) { return true; }
        
        if (!"O".equals(tag) && !".".equals(tag) && !prevType.equals(type)) {
            return true;
        }

        // conlleval script treats these as having length 1
        if ("]".equals(tag)) { return true; }
        if ("[".equals(tag)) { return true; }
        if ("S".equals(tag)) { return true; } // This singleton is not included in conlleval.
        if ("U".equals(tag)) { return true; } // This singleton is not included in conlleval.
        
        return false;
    }

    /** Returns true iff the a chunk ended between the current and previous token. */
    private static boolean endOfChunk(String prevTag, String tag, String prevType, String type) {
        if ("B".equals(prevTag) && "B".equals(tag)) { return true; }
        if ("B".equals(prevTag) && "O".equals(tag)) { return true; }
        if ("I".equals(prevTag) && "B".equals(tag)) { return true; }
        if ("I".equals(prevTag) && "O".equals(tag)) { return true; }
        
        if ("E".equals(prevTag) && "E".equals(tag)) { return true; }
        if ("E".equals(prevTag) && "I".equals(tag)) { return true; }
        if ("E".equals(prevTag) && "O".equals(tag)) { return true; }
        if ("I".equals(prevTag) && "O".equals(tag)) { return true; }
        
        if (!"O".equals(prevTag) && !".".equals(prevTag) && !prevType.equals(type)) {
            return true;
        }

        // conlleval script treats these as having length 1
        if ("]".equals(prevTag)) { return true; }
        if ("[".equals(prevTag)) { return true; }
        if ("S".equals(prevTag)) { return true; } // This singleton is not included in conlleval.
        if ("U".equals(prevTag)) { return true; } // This singleton is not included in conlleval.
        
        return false;
    }

    @Override
    protected boolean isNilLabel(String label) {
        return label == null;
    }
    
    @Override
    protected String getDataType() {
        return "NER";
    }

    @Override
    protected List<String> getLabels(AnnoSentence sent) {
        throw new RuntimeException("never called");
    }
    
}
