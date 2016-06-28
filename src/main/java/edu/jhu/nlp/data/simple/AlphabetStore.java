package edu.jhu.nlp.data.simple;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map.Entry;
import java.util.Set;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.pacaya.util.collections.QLists;
import edu.jhu.prim.Primitives;
import edu.jhu.prim.Primitives.MutableInt;
import edu.jhu.prim.bimap.CountingIntObjectBimap;
import edu.jhu.prim.bimap.IntObjectBimap;
import edu.jhu.prim.list.IntArrayList;
import edu.jhu.prim.tuple.ComparablePair;
import edu.jhu.prim.tuple.Pair;
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
    IntObjectBimap<String> lcWords;
    IntObjectBimap<String> prefixes;
    IntObjectBimap<String> suffixes;
    IntObjectBimap<String> lemmas;
    IntObjectBimap<String> posTags;
    IntObjectBimap<String> cposTags;
    IntObjectBimap<String> clusters;
    IntObjectBimap<String> clusterPrefixes;
    IntObjectBimap<String> feats;
    IntObjectBimap<String> deprels;
    IntObjectBimap<String> srlPredSenses;
    IntObjectBimap<String> srlArgs;
    private List<IntObjectBimap<String>> as;

    private int wordTopNCutoff;
    
    public AlphabetStore(Iterable<AnnoSentence> sents) {
        // The string generators for prefixes and suffixes create all affixes up to a given max
        // length. The string to int mapping is kept in only a single IntObjectBimap.
        MultiStrGetter prefixGetter = new MultiStrGetter(
                IntStream.range(0, maxPrefixLen).mapToObj(
                        i -> new AffixGetter(i+1, true, wordGetter)));
        MultiStrGetter suffixGetter = new MultiStrGetter(
                IntStream.range(0, maxSuffixLen).mapToObj(
                        i -> new AffixGetter(i+1, false, wordGetter)));
        MultiStrGetter clusterPrefixGetter = new MultiStrGetter(
                IntStream.range(0, maxClusterPrefixLen).mapToObj(
                        i -> new AffixGetter(i+1, true, clusterGetter)));
        
        words = getAlphabet("word", wordGetter, MAX_WORD, sents);
        lcWords = getAlphabet("lcWord", lcWordGetter, MAX_WORD, sents);
        prefixes = getAlphabet("prefix", prefixGetter, MAX_PREFIX, sents);
        suffixes = getAlphabet("suffix", suffixGetter, MAX_SUFFIX, sents);
        lemmas = getAlphabet("lemma", lemmaGetter, MAX_LEMMA, sents);
        posTags = getAlphabet("pos", posTagGetter, MAX_POS, sents);
        cposTags = getAlphabet("cpos", cposTagGetter, MAX_CPOS, sents);
        clusters = getAlphabet("cluster", clusterGetter, MAX_CLUSTER, sents);
        clusterPrefixes = getAlphabet("clusterPrefix", clusterPrefixGetter, MAX_CLUSTER, sents);
        feats = getAlphabet("feat", featGetter, MAX_FEAT, sents);
        deprels= getAlphabet("deprel", deprelGetter, MAX_DEPREL, sents);
        srlPredSenses = getAlphabet("srlPredSense", srlPredSenseGetter, MAX_SRL_PRED_SENSE, sents);
        srlArgs = getAlphabet("srlArgs", srlArgGetter, MAX_SRL_ARG, sents);
        
        // Compute the minimum frequence of the top 800 most frequent words.
        wordTopNCutoff = getTopNCutoff(words, 800);
        
        as = QLists.getList(words, lcWords, prefixes, suffixes, lemmas, posTags, cposTags, 
                clusters, clusterPrefixes, feats, deprels, srlPredSenses, srlArgs);
        this.stopGrowth();
    }

    /**
     * Gets a mapping from ints to strings. Types occurring fewer than K times are re-mapped to UNK,
     * where K is the minimum value such that the maximum index of the final mapping is
     * less-than-or-equal to maxIdx.
     */
    private static CountingIntObjectBimap<String> getAlphabet(String name, StrApplier sa, int maxIdx, Iterable<AnnoSentence> sents) {
        CountingIntObjectBimap<String> counter = countStrings(sa, sents);
        CountingIntObjectBimap<String> alphabet = applyCountCutoffToGetAlphabet(name, maxIdx, counter);
        return alphabet;
    }

    /** Gets a mapping from ints to strings, with counts of the number of times each one was observed. */
    private static CountingIntObjectBimap<String> countStrings(StrApplier sa, Iterable<AnnoSentence> sents) {
        CountingIntObjectBimap<String> counter = new CountingIntObjectBimap<>();
        for (AnnoSentence sent : sents) {
            sa.apply(sent, (str) -> counter.lookupIndex(str));
        }
        return counter;
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

    public interface StrApplier extends Serializable {
        void apply(AnnoSentence sent, StrCounter sc);
    }
    
    public interface StrCounter {
        void count(String str);
    }
    
    private static abstract class StrGetter implements StrApplier {
        
        private static final long serialVersionUID = 1L;

        @Override
        public void apply(AnnoSentence sent, StrCounter sc) {
            List<String> strs = getStrs(sent);
            if (strs == null) { return; }
            for (String str : strs) {
                sc.count(str);
            }
        }
        
        public abstract List<String> getStrs(AnnoSentence sent);
    }
    
    /** For each token, get all affixes up to a maximum length. */
    public static class AffixGetter extends StrGetter {
        private static final long serialVersionUID = 1L;
        private int max;
        private boolean isPre;
        private StrGetter getter;
        public AffixGetter(int max, boolean isPre, StrGetter getter) { 
            this.max = max;
            this.isPre = isPre;
            this.getter = getter;
        }
        public List<String> getStrs(AnnoSentence sent) { 
            List<String> input = getter.getStrs(sent);
            if (input == null) { return Collections.emptyList(); }
            ArrayList<String> strs = new ArrayList<>(input.size());
            for (int i=0; i<input.size(); i++) {
                String s = input.get(i);
                s = getAffix(s, max, isPre);
                strs.add(s);
            }
            return strs;
        }
        public static String getAffix(String s, int max, boolean isPre) {
            if (isPre) {
                s = s.substring(0, Math.min(s.length(), max)); // prefix
            } else {
                s = s.substring(Math.max(0, s.length() - max), s.length()); // suffix
            }
            return s;
        }
    }
    
    /** Concatenates the output of multiple StrGetters. */
    private static class MultiStrGetter extends StrGetter {
        private static final long serialVersionUID = 1L;
        List<StrGetter> getters = new ArrayList<>();
        public MultiStrGetter(StrGetter... getters) {
            for (StrGetter g : getters) {
                this.getters.add(g);
            }
        }
        public MultiStrGetter(Stream<StrGetter> getters) {
            getters.forEach(g -> this.getters.add(g));
        }
        @Override
        public List<String> getStrs(AnnoSentence sent) {
            ArrayList<String> strs = new ArrayList<>();
            for (StrGetter g : getters) {
                strs.addAll(g.getStrs(sent));
            }
            return strs;
        }
    }
    
    private StrGetter wordGetter = new StrGetter() {
        private static final long serialVersionUID = 1L;
        public List<String> getStrs(AnnoSentence sent) { return sent.getWords(); }
    };
    private StrGetter lcWordGetter = new StrGetter() {
        private static final long serialVersionUID = 1L;
        public List<String> getStrs(AnnoSentence sent) { return sent.getLowerCaseWords(); }
    };
    private StrGetter lemmaGetter = new StrGetter() {
        private static final long serialVersionUID = 1L;
        public List<String> getStrs(AnnoSentence sent) { return sent.getLemmas(); }
    };
    private StrGetter posTagGetter = new StrGetter() {
        private static final long serialVersionUID = 1L;
        public List<String> getStrs(AnnoSentence sent) { return sent.getPosTags(); }
    };
    private StrGetter cposTagGetter = new StrGetter() { 
        private static final long serialVersionUID = 1L;
        public List<String> getStrs(AnnoSentence sent) { return sent.getCposTags(); }
    };
    private StrGetter clusterGetter = new StrGetter() {
        private static final long serialVersionUID = 1L;
        public List<String> getStrs(AnnoSentence sent) { return sent.getClusters(); }
    };
    private StrGetter featGetter = new StrGetter() {
        private static final long serialVersionUID = 1L;
        public List<String> getStrs(AnnoSentence sent) {
            if (sent.getFeats() == null) { return null; }
            ArrayList<String> strs = new ArrayList<>();
            for (List<String> featList : sent.getFeats()) {
                if (featList != null) {
                    strs.addAll(featList);
                }
            }
            return strs;
        }
    };
    private StrGetter deprelGetter = new StrGetter() {
        private static final long serialVersionUID = 1L;
        public List<String> getStrs(AnnoSentence sent) { return sent.getDeprels(); }
    };
    private StrApplier srlPredSenseGetter = new StrApplier() {
        private static final long serialVersionUID = 1L;
        public void apply(AnnoSentence sent, StrCounter sc) {
            if (sent.getSrlGraph() != null) {
                for (int p=0; p<sent.getSrlGraph().size(); p++) {
                    sc.count(sent.getSrlGraph().get(-1, p));
                }
            }
        }
    };
    private StrApplier srlArgGetter = new StrApplier() {
        private static final long serialVersionUID = 1L;
        public void apply(AnnoSentence sent, StrCounter sc) {
            if (sent.getSrlGraph() != null) {
                for (int p=0; p<sent.getSrlGraph().size(); p++) {
                    for (int c=0; c<sent.getSrlGraph().size(); c++) {
                        sc.count(sent.getSrlGraph().get(p, c));
                    }
                }
            }
        }
    };

    public int getWordTopNCutoff() {
        return wordTopNCutoff;
    }
    
}
