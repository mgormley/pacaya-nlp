package edu.jhu.nlp.data.simple;

import java.util.List;

import edu.jhu.nlp.data.DepGraph;
import edu.jhu.nlp.data.simple.AlphabetStore.AffixGetter;
import edu.jhu.nlp.features.FeaturizedToken;
import edu.jhu.nlp.tag.StrictPosTagAnnotator.StrictPosTag;
import edu.jhu.prim.bimap.IntObjectBimap;
import edu.jhu.prim.list.ByteArrayList;
import edu.jhu.prim.list.IntArrayList;
import edu.jhu.prim.list.ShortArrayList;
import edu.jhu.prim.util.SafeCast;

public class IntAnnoSentence {

    private ShortArrayList words;
    private ShortArrayList lcWords; // lower-case words
    private ShortArrayList[] prefixes;
    private ShortArrayList[] suffixes;
    private boolean[] isCapitalized;    
    private ShortArrayList lemmas;
    private ByteArrayList posTags;
    private ByteArrayList cposTags;
    private ByteArrayList coarserPosTags;
    private ShortArrayList clusters;
    private ShortArrayList[] clusterPrefixes;
    private ShortArrayList[] feats;
    private ByteArrayList deprels;
    private ShortArrayList srlPredSenses;
    private ByteArrayList srlArgs;
    // TODO: private IntNaryTree naryTree;
    
    private ShortArrayList numVerbsToLeft;
    private ShortArrayList numNounsToLeft;
    private ShortArrayList numPuncsToLeft;
    private ShortArrayList numConjsToLeft;
    
    private AnnoSentence sent;
    private AlphabetStore store;
    
    public IntAnnoSentence(AnnoSentence sent, AlphabetStore store) {
        this.sent = sent;
        this.store = store;
        this.words = getShorts(sent.getWords(), store.words);
        this.lcWords = getShorts(sent.getLowerCaseWords(), store.lcWords);
        this.prefixes = getAffixShorts(sent.getWords(), store.prefixes, store.maxPrefixLen, true);
        this.suffixes = getAffixShorts(sent.getWords(), store.suffixes, store.maxSuffixLen, false);
        this.isCapitalized = getIsCapitalized(sent.getWords());
        this.lemmas = getShorts(sent.getLemmas(), store.lemmas);
        this.posTags = getBytes(sent.getPosTags(), store.posTags);
        this.cposTags = getBytes(sent.getCposTags(), store.cposTags);
        this.clusters = getShorts(sent.getClusters(), store.clusters);
        this.clusterPrefixes = getAffixShorts(sent.getClusters(), store.clusterPrefixes, store.maxClusterPrefixLen, true);
        if (sent.getFeats() != null) {
            this.feats = new ShortArrayList[sent.getFeats().size()];
            for (int i=0; i<feats.length; i++) {
                this.feats[i] = getShorts(sent.getFeats(i), store.feats);
            }
        }
        this.deprels = getBytes(sent.getDeprels(), store.deprels);
        if (sent.getSrlGraph() != null) {
            this.srlPredSenses = getShorts(sent.getPredSenses(), store.srlPredSenses);
            this.srlArgs = getBytesFromDepGraph(sent.getSrlGraph(), store.srlArgs);
        }
        if (StrictPosTag.values().length > AlphabetStore.MAX_STRICT_POS) {
            throw new IllegalStateException("Too many strict POS tags.");
        }
        this.coarserPosTags = getBytesFromEnums(sent.getStrictPosTags());
        this.numVerbsToLeft = getNumToLeft(sent.getStrictPosTags(), StrictPosTag.VERB);
        this.numNounsToLeft = getNumToLeft(sent.getStrictPosTags(), StrictPosTag.NOUN);
        this.numPuncsToLeft = getNumToLeft(sent.getStrictPosTags(), StrictPosTag.PUNC);
        this.numConjsToLeft = getNumToLeft(sent.getStrictPosTags(), StrictPosTag.CONJ);
    }

    private static IntArrayList getInts(List<String> tokens, IntObjectBimap<String> alphabet) {
        if (tokens == null) { return null; }
        IntArrayList arr = new IntArrayList(tokens.size());
        for (int i=0; i<tokens.size(); i++) {
            int idx = AlphabetStore.safeLookup(alphabet, tokens.get(i));
            arr.add(idx);
        }
        return arr;
    }
    
    private static ShortArrayList getShorts(List<String> tokens, IntObjectBimap<String> alphabet) {
        if (tokens == null) { return null; }
        ShortArrayList arr = new ShortArrayList(tokens.size());
        for (int i=0; i<tokens.size(); i++) {
            int idx = AlphabetStore.safeLookup(alphabet, tokens.get(i));
            arr.add(SafeCast.safeIntToUnsignedShort(idx));
        }
        return arr;
    }
    
    private static ByteArrayList getBytes(List<String> tokens, IntObjectBimap<String> alphabet) {
        if (tokens == null) { return null; }
        ByteArrayList arr = new ByteArrayList(tokens.size());
        for (int i=0; i<tokens.size(); i++) {
            int idx = AlphabetStore.safeLookup(alphabet, tokens.get(i));
            arr.add(SafeCast.safeIntToUnsignedByte(idx));
        }
        return arr;
    }
    
    private static ByteArrayList getBytesFromEnums(List<? extends Enum<?>> tokens) {
        if (tokens == null) { return null; }
        ByteArrayList arr = new ByteArrayList(tokens.size());
        for (int i=0; i<tokens.size(); i++) {
            int idx = tokens.get(i).ordinal();
            arr.add(SafeCast.safeIntToUnsignedByte(idx));
        }
        return arr;
    }

    private static <X> ShortArrayList getNumToLeft(List<X> tokens, X type) {
        if (tokens == null) { return null; }
        ShortArrayList arr = new ShortArrayList(tokens.size());
        int numSeen = 0;
        for (int i=0; i<=tokens.size(); i++) {
            arr.add(SafeCast.safeIntToShort(numSeen));
            if (i < tokens.size() && (type == tokens.get(i) || type.equals(tokens.get(i)))) {
                numSeen++;
            }
        }
        return arr;
    }
    
    private static ShortArrayList[] getAffixShorts(List<String> tokens, IntObjectBimap<String> alphabet, int maxLen, boolean isPre) {
        if (tokens == null) { return null; }
        // TODO: Simpler? short[][] arr2d = new short[maxLen][tokens.size()];
        ShortArrayList[] arr = new ShortArrayList[maxLen];
        for (int k=0; k<maxLen; k++) {
            arr[k] = new ShortArrayList(tokens.size());
            for (int i=0; i<tokens.size(); i++) {
                int idx = AlphabetStore.safeLookup(alphabet, AffixGetter.getAffix(tokens.get(i), k+1, isPre));
                arr[k].add(SafeCast.safeIntToUnsignedShort(idx));
            }
        }
        return arr;
    }

    private static boolean[] getIsCapitalized(List<String> words) {
        boolean[] isCapitalized = new boolean[words.size()];
        for (int i=0; i<words.size(); i++) {
            isCapitalized[i] = FeaturizedToken.capitalized(words.get(i));
        }
        return isCapitalized;
    }

    /** Gets all edges except for (-1, c) edges. */
    private static ByteArrayList getBytesFromDepGraph(DepGraph graph, IntObjectBimap<String> alphabet) {
        if (graph == null) { return null; }
        int n = graph.size();
        ByteArrayList edges = new ByteArrayList(n*n);
        for (int p=0; p<n; p++) {
            for (int c=0; c<n; c++) {
                int idx = AlphabetStore.safeLookup(alphabet, graph.get(p, c));
                // Adding to position p*n + c
                edges.add(SafeCast.safeIntToUnsignedByte(idx));
            }
        }
        return edges;
    }
    
    /** Gets the i'th word. */
    public short getWord(int i) {
        return words.get(i);
    }
    
    /** Gets the i'th lowercased word. */
    public short getLcWord(int i) {
        return lcWords.get(i);
    }

    /** Gets the i'th prefix of length len. */
    public short getPrefix(int i, int len) {
        return prefixes[len-1].get(i);
    }
    
    /** Gets the i'th suffix of length len. */
    public short getSuffix(int i, int len) {
        return suffixes[len-1].get(i);
    }
    
    /** Gets whether the i'th word is capitalized. */
    public boolean isCapitalized(int i) {
        return isCapitalized[i];
    }
        
    /** Gets the i'th lemma. */
    public short getLemma(int i) {
        return lemmas.get(i);
    }

    /** Gets the i'th POS tag. */
    public byte getPosTag(int i) {
        return posTags.get(i);
    }
    
    /** Gets the i'th Coarse POS tag. */
    public byte getCposTag(int i) {
        return cposTags.get(i);
    }
    
    /** Gets the i'th Strict POS tag. */
    public byte getStrictPosTag(int i) {
        return coarserPosTags.get(i);
    }

    /** Gets the i'th Distributional Similarity Cluster ID. */
    public short getCluster(int i) {
        return clusters.get(i);
    }

    /** Gets the i'th cluster prefix of length len. */
    public short getClusterPrefix(int i, int len) {
        return clusterPrefixes[len-1].get(i);
    }
    
    /** Gets the features (e.g. morphological features) of the i'th word. */
    public ShortArrayList getFeats(int i) {
        return feats[i];
    }

    /** Gets the dependency relation label for the arc from the i'th word to its parent. */
    public byte getDeprel(int i) {
        return deprels.get(i);
    }

    /** Gets the predicate sense of the i'th token. */
    public short getSrlPredSense(int i) {
        return srlPredSenses.get(i);
    }

    /** Gets the semantic argument label from token p to token c. */
    public byte getSrlArg(int p, int c) {
        return srlArgs.get(p * this.size() + c);
    }
    
    /** Gets the number of verbs in between tokens a and b. */
    public short getNumVerbsInBetween(int a, int b) {
        return getNumInBetween(numVerbsToLeft, a, b);
    }

    /** Gets the number of nouns in between tokens a and b. */
    public short getNumNounsInBetween(int a, int b) {
        return getNumInBetween(numNounsToLeft, a, b);
    }

    /** Gets the number of punctuations in between tokens a and b. */
    public short getNumPuncsInBetween(int a, int b) {
        return getNumInBetween(numPuncsToLeft, a, b);
    }

    /** Gets the number of conjunctions in between tokens a and b. */
    public short getNumConjsInBetween(int a, int b) {
        return getNumInBetween(numConjsToLeft, a, b);
    }
    
    private static short getNumInBetween(ShortArrayList numToLeft, int a, int b) {
        if (b > a) {
            return SafeCast.safeIntToShort(numToLeft.get(b) - numToLeft.get(a+1));
        } else if (a > b) {
            return SafeCast.safeIntToShort(numToLeft.get(a) - numToLeft.get(b+1));
        } else {
            // a == b
            return 0;
        }
    }

    public int size() {
        return words.size();
    }
    
    public AnnoSentence getAnnoSentence() {
        return sent;
    }
    
    public AlphabetStore getStore() {
        return store;
    }

}
