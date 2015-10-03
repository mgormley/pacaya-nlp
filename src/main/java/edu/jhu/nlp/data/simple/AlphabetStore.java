package edu.jhu.nlp.data.simple;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.pacaya.util.collections.QLists;
import edu.jhu.prim.bimap.CountingIntObjectBimap;
import edu.jhu.prim.bimap.IntObjectBimap;

public class AlphabetStore implements Serializable {

    private static final long serialVersionUID = 1L;

    private static final Logger log = LoggerFactory.getLogger(AlphabetStore.class);
    
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
    
    IntObjectBimap<String> words;
    IntObjectBimap<String> prefixes;
    IntObjectBimap<String> suffixes;
    IntObjectBimap<String> lemmas;
    IntObjectBimap<String> posTags;
    IntObjectBimap<String> cposTags;
    IntObjectBimap<String> clusters;
    IntObjectBimap<String> feats;
    IntObjectBimap<String> deprels;
    // TODO: 
    //Alphabet<String> lexAlphabet;
    //Alphabet<String> ntAlphabet;    
    private List<IntObjectBimap<String>> as;
    
    public final int maxPrefixLen = 5;
    public final int maxSuffixLen = 5;
    
    public AlphabetStore(Iterable<AnnoSentence> sents) {
        // The string generators for prefixes and suffixes create all affixes up to a given max
        // length. The string to int mapping is kept in only a single IntObjectBimap.
        MultiStrGetter prefixGetter = new MultiStrGetter(
                IntStream.range(0, maxPrefixLen).mapToObj(
                        i -> new AffixGetter(i+1, true)));
        MultiStrGetter suffixGetter = new MultiStrGetter(
                IntStream.range(0, maxSuffixLen).mapToObj(
                        i -> new AffixGetter(i+1, false)));
        
        words = getInitAlphabet("word", wordGetter, IntAnnoSentence.MAX_WORD, sents);
        prefixes = getInitAlphabet("prefix", prefixGetter, IntAnnoSentence.MAX_PREFIX, sents);
        suffixes = getInitAlphabet("suffix", suffixGetter, IntAnnoSentence.MAX_SUFFIX, sents);
        lemmas = getInitAlphabet("lemma", lemmaGetter, IntAnnoSentence.MAX_LEMMA, sents);
        posTags = getInitAlphabet("pos", posTagGetter, IntAnnoSentence.MAX_POS, sents);
        cposTags = getInitAlphabet("cpos", cposTagGetter, IntAnnoSentence.MAX_CPOS, sents);
        clusters = getInitAlphabet("cluster", clusterGetter, IntAnnoSentence.MAX_CLUSTER, sents);
        feats = getInitAlphabet("feat", featGetter, IntAnnoSentence.MAX_FEAT, sents);
        deprels= getInitAlphabet("deprel", deprelGetter, IntAnnoSentence.MAX_DEPREL, sents);
        
        as = QLists.getList(words, prefixes, suffixes, lemmas, posTags, cposTags, clusters, feats, deprels);
        this.stopGrowth();
    }

    private static IntObjectBimap<String> getInitAlphabet(String name, StrGetter sg, int maxIdx, Iterable<AnnoSentence> sents) {
        CountingIntObjectBimap<String> counter = new CountingIntObjectBimap<>();
        for (AnnoSentence sent : sents) {
            List<String> strs = sg.getStrs(sent);
            if (strs != null) {
                for (String str : strs) {
                    counter.lookupIndex(str);
                }
            }
        }
        IntObjectBimap<String> alphabet;
        for (int cutoff = 1; ; cutoff++) {
            alphabet = getInitAlphabet();
            for (int idx=0; idx<counter.size(); idx++) {
                String str = counter.lookupObject(idx);
                int count = counter.lookupObjectCount(idx);
                if (count >= cutoff) {
                    alphabet.lookupIndex(str);
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
    
    private static IntObjectBimap<String> getInitAlphabet() {
        IntObjectBimap<String> alphabet = new IntObjectBimap<String>();
        //for (SpecialToken tok : SpecialToken.values()) {
        for (int i=0; i<NUM_SPECIAL_TOKS; i++) {
            int idx = alphabet.lookupIndex(specialTokenStrs[i]);
            if (idx != i) {
                throw new RuntimeException("Expecting first index from alphabet to be 0");
            }
        }
        return alphabet;
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
    
    public int getWordIdx(String word) {
        return safeLookup(words, word);
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

    public int getFeatIdx(String feat) {
        return safeLookup(feats, feat);
    }

    public int getDeprelIdx(String deprel) {
        return safeLookup(deprels, deprel);
    }
    

    public interface StrGetter extends Serializable {
        List<String> getStrs(AnnoSentence sent);
    }
    public static class AffixGetter implements StrGetter {
        private static final long serialVersionUID = 1L;
        private int max;
        private boolean isPre;
        public AffixGetter(int max, boolean isPre) { 
            this.max = max;
            this.isPre = isPre;
        }
        public List<String> getStrs(AnnoSentence sent) { 
            ArrayList<String> strs = new ArrayList<>(sent.size());
            for (int i=0; i<sent.size(); i++) {
                String s = sent.getWord(i);
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
    private static class MultiStrGetter implements StrGetter {
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
    
}
