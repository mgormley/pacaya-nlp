package edu.jhu.nlp.data.simple;

import java.io.Serializable;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.pacaya.util.collections.QLists;
import edu.jhu.prim.Primitives;
import edu.jhu.prim.bimap.CountingIntObjectBimap;
import edu.jhu.prim.bimap.IntObjectBimap;
import edu.jhu.prim.list.IntArrayList;
import edu.jhu.prim.util.SafeCast;

public class AlphabetStore implements Serializable {
    
    private static final long serialVersionUID = 1L;

    private static final Logger log = LoggerFactory.getLogger(AlphabetStore.class);
    
    // Tunable parameters.
    public final int maxPrefixLen = 5;
    public final int maxSuffixLen = 5;
    public final int maxClusterPrefixLen = 5;
    
    // Special Tokens.
    public static final int NUM_SPECIAL_TOKS = 4;
    public static final String TOK_UNK_STR = "TOK_UNK";
    public static final String TOK_START_STR = "TOK_START";
    public static final String TOK_END_STR = "TOK_END";
    public static final String TOK_WALL_STR = "TOK_WALL";
    public static final int TOK_UNK_INT = 0;
    public static final int TOK_START_INT = 1;
    public static final int TOK_END_INT = 2;
    public static final int TOK_WALL_INT = 3;
    public static String[] specialTokenStrs = new String[] { TOK_UNK_STR, TOK_START_STR, TOK_END_STR, TOK_WALL_STR};

    // Maximum values (in integer space) for the various types of strings.
    //
    // We reserve the value -1 for truly unknown values.
    private static final int BYTE_MAX_IDX = Primitives.INT_MAX_UBYTE-1;
    private static final int SHORT_MAX_IDX = Primitives.INT_MAX_USHORT-1;
    // TODO: We're missing a bit because our Alphabets always return signed values. 
    private static final int INT_MAX_IDX = Integer.MAX_VALUE; // This is NOT 0xffffffffL;

    final static int MAX_WORD = SHORT_MAX_IDX;
    final static int MAX_PREFIX = SHORT_MAX_IDX;
    final static int MAX_SUFFIX = SHORT_MAX_IDX;
    final static int MAX_LEMMA = SHORT_MAX_IDX;
    final static int MAX_POS = BYTE_MAX_IDX;
    final static int MAX_CPOS = BYTE_MAX_IDX;
    final static int MAX_STRICT_POS = BYTE_MAX_IDX;
    final static int MAX_CLUSTER = SHORT_MAX_IDX;
    final static int MAX_FEAT = SHORT_MAX_IDX;
    final static int MAX_DEPREL = BYTE_MAX_IDX;
    final static int MAX_SRL_PRED_SENSE = SHORT_MAX_IDX;
    final static int MAX_SRL_ARG = BYTE_MAX_IDX;
    
    CountingIntObjectBimap<String> words;
    CountingIntObjectBimap<String> lcWords;
    CountingIntObjectBimap<String> prefixes;
    CountingIntObjectBimap<String> suffixes;
    CountingIntObjectBimap<String> lemmas;
    CountingIntObjectBimap<String> posTags;
    CountingIntObjectBimap<String> cposTags;
    CountingIntObjectBimap<String> clusters;
    CountingIntObjectBimap<String> clusterPrefixes;
    CountingIntObjectBimap<String> feats;
    CountingIntObjectBimap<String> deprels;
    CountingIntObjectBimap<String> srlPredSenses;
    CountingIntObjectBimap<String> srlArgs;
    
    private final List<IntObjectBimap<String>> as;
    
    private final int wordTopNCutoff;
    
    private static class CountingIntObjectBimapWithoutIdxLookup<T> extends CountingIntObjectBimap<T> {

        private static final long serialVersionUID = 1L;

        @Override
        public int lookupIndex(T object, boolean addIfMissing) {
            super.lookupIndex(object, addIfMissing);
            return -1;
        }
        
    }
    
    public AlphabetStore(Iterable<AnnoSentence> sents) {
        words = new CountingIntObjectBimapWithoutIdxLookup<>();
        lcWords = new CountingIntObjectBimapWithoutIdxLookup<>();
        prefixes = new CountingIntObjectBimapWithoutIdxLookup<>();
        suffixes = new CountingIntObjectBimapWithoutIdxLookup<>();
        lemmas = new CountingIntObjectBimapWithoutIdxLookup<>();
        posTags = new CountingIntObjectBimapWithoutIdxLookup<>();
        cposTags = new CountingIntObjectBimapWithoutIdxLookup<>();
        clusters = new CountingIntObjectBimapWithoutIdxLookup<>();
        clusterPrefixes = new CountingIntObjectBimapWithoutIdxLookup<>();
        feats = new CountingIntObjectBimapWithoutIdxLookup<>();
        deprels= new CountingIntObjectBimapWithoutIdxLookup<>();
        srlPredSenses = new CountingIntObjectBimapWithoutIdxLookup<>();
        srlArgs = new CountingIntObjectBimapWithoutIdxLookup<>();
        
        // Construct the IntAnnoSentences only for counting.
        for (AnnoSentence sent : sents) {
            new IntAnnoSentence(sent, this);
        }

        // Apply the count cutoffs.
        words = applyCountCutoffToGetAlphabet("word", MAX_WORD, words);
        lcWords = applyCountCutoffToGetAlphabet("lcWord", MAX_WORD, lcWords);
        prefixes = applyCountCutoffToGetAlphabet("prefix", MAX_PREFIX, prefixes);
        suffixes = applyCountCutoffToGetAlphabet("suffix", MAX_SUFFIX, suffixes);
        lemmas = applyCountCutoffToGetAlphabet("lemma", MAX_LEMMA, lemmas);
        posTags = applyCountCutoffToGetAlphabet("pos", MAX_POS, posTags);
        cposTags = applyCountCutoffToGetAlphabet("cpos", MAX_CPOS, cposTags);
        clusters = applyCountCutoffToGetAlphabet("cluster", MAX_CLUSTER, clusters);
        clusterPrefixes = applyCountCutoffToGetAlphabet("clusterPrefix", MAX_CLUSTER, clusterPrefixes);
        feats = applyCountCutoffToGetAlphabet("feat", MAX_FEAT, feats);
        deprels= applyCountCutoffToGetAlphabet("deprel", MAX_DEPREL, deprels);
        srlPredSenses = applyCountCutoffToGetAlphabet("srlPredSense", MAX_SRL_PRED_SENSE, srlPredSenses);
        srlArgs = applyCountCutoffToGetAlphabet("srlArgs", MAX_SRL_ARG, srlArgs);
        
        // Compute the minimum frequence of the top 800 most frequent words.
        wordTopNCutoff = getTopNCutoff(words, 800);
        
        as = QLists.getList(words, lcWords, prefixes, suffixes, lemmas, posTags, cposTags, 
                clusters, clusterPrefixes, feats, deprels, srlPredSenses, srlArgs);
        this.stopGrowth();
    }
    
    /**
     * Transforms a mapping from ints to strings (with counts!) to ensure it does not exceed a
     * maximum size. Types occurring fewer than K times are re-mapped to UNK, where K is the minimum
     * value such that the maximum index of the final mapping is less-than-or-equal to maxIdx.
     */
    protected static CountingIntObjectBimap<String> applyCountCutoffToGetAlphabet(String name, int maxIdx,
            CountingIntObjectBimap<String> counter) {
        // Apply count-cutoffs, increasing K (the cutoff) until the total number of types is <= maxIdx.
        CountingIntObjectBimap<String> alphabet;
        for (int cutoff = 1; ; cutoff++) {
            alphabet = getInitAlphabet();
            alphabet.setObjectCount(TOK_START_INT, 0);
            alphabet.setObjectCount(TOK_END_INT, 0);
            alphabet.setObjectCount(TOK_WALL_INT, 0);
            alphabet.setObjectCount(TOK_UNK_INT, 0);
            for (int idx=0; idx<counter.size(); idx++) {
                String str = counter.lookupObject(idx);
                int count = counter.lookupObjectCount(idx);
                if (count >= cutoff) {
                    int newIdx = alphabet.lookupIndex(str);
                    alphabet.setObjectCount(newIdx, count);
                } else if (idx >= NUM_SPECIAL_TOKS) {
                    count += alphabet.lookupObjectCount(TOK_UNK_INT);
                    alphabet.setObjectCount(TOK_UNK_INT, count);
                }
            }
            if (alphabet.size()-1 <= maxIdx) {
                log.info(String.format("For %s: Type count = %d Alphabet count = %d Cutoff = %d", 
                        name, counter.size(), alphabet.size(), cutoff));
                break;
            }
        }
        return alphabet;
    }

    /**
     * Gets a mapping from ints to strings, which is initialized with the special tokens occupying
     * their reserved positions.
     */
    private static CountingIntObjectBimap<String> getInitAlphabet() {
        CountingIntObjectBimap<String> alphabet = new CountingIntObjectBimap<String>();
        //for (SpecialToken tok : SpecialToken.values()) {
        for (int i=0; i<NUM_SPECIAL_TOKS; i++) {
            int idx = alphabet.lookupIndex(specialTokenStrs[i]);
            if (idx != i) {
                throw new RuntimeException("Expecting first index from alphabet to be 0");
            }
        }
        assert alphabet.lookupIndex(TOK_UNK_STR) == TOK_UNK_INT;
        assert alphabet.lookupIndex(TOK_START_STR) == TOK_START_INT;
        assert alphabet.lookupIndex(TOK_END_STR) == TOK_END_INT;
        assert alphabet.lookupIndex(TOK_WALL_STR) == TOK_WALL_INT;
        return alphabet;
    }

    /** Gets the frequency of the topN'th most frequent word. */
    private static int getTopNCutoff(CountingIntObjectBimap<String> words, int topN) {
        IntArrayList idxCountMap = new IntArrayList(words.getInternalIdxCountMap());
        idxCountMap.sortDesc();
        int i = Math.min(idxCountMap.size() - 1, topN);
        int cutoff = idxCountMap.get(i);
        while(i > 0 && cutoff == 0) {
            cutoff = idxCountMap.get(--i);
        }
        return cutoff;
    }

    public void startGrowth() {
        for (IntObjectBimap<String> a : as) {
            a.startGrowth();
        }
    }
    
    public void stopGrowth() {
        for (IntObjectBimap<String> a : as) {
            a.stopGrowth();
        }
    }
    
    static int safeLookup(IntObjectBimap<String> alphabet, String tokStr) {
        int idx = alphabet.lookupIndex(tokStr);
        if (idx == -1) {
            idx = TOK_UNK_INT;
        }
        return idx;
    }

    public int getWordTypeCount(short wordIdx) {
        assert MAX_WORD == SHORT_MAX_IDX;
        return words.lookupObjectCount(SafeCast.safeUnsignedShortToInt(wordIdx));
    }
    
    public short getWordIdx(String word) {
        return SafeCast.safeIntToUnsignedShort(safeLookup(words, word));
    }

    public int getLcWordIdx(String lcWord) {
        return SafeCast.safeIntToUnsignedShort(safeLookup(lcWords, lcWord));
    }

    public int getPrefixIdx(String prefix) {
        return safeLookup(prefixes, prefix);
    }
    
    public int getSuffixIdx(String suffix) {
        return safeLookup(suffixes, suffix);
    }
    
    public int getLemmaIdx(String lemma) {
        return safeLookup(lemmas, lemma);
    }

    public int getPosTagIdx(String pos) {
        return safeLookup(posTags, pos);
    }

    public int getCposTagIdx(String cpos) {
        return safeLookup(cposTags, cpos);
    }

    public int getClusterIdx(String cluster) {
        return safeLookup(clusters, cluster);
    }
    
    public int getClusterPrefixIdx(String clusterPrefix) {
        return safeLookup(clusterPrefixes, clusterPrefix);
    }

    public int getFeatIdx(String feat) {
        return safeLookup(feats, feat);
    }

    public int getDeprelIdx(String deprel) {
        return safeLookup(deprels, deprel);
    }

    public int getSrlPredSenseIdx(String sense) {
        return safeLookup(srlPredSenses, sense);
    }
    
    public int getSrlArgIdx(String arg) {
        return safeLookup(srlArgs, arg);
    }
    
    public int getWordTopNCutoff() {
        return wordTopNCutoff;
    }
    
}
