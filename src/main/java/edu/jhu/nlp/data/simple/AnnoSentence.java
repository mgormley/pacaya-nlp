package edu.jhu.nlp.data.simple;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;

import edu.jhu.nlp.data.DepEdgeMask;
import edu.jhu.nlp.data.DepGraph;
import edu.jhu.nlp.data.NerMention;
import edu.jhu.nlp.data.NerMentions;
import edu.jhu.nlp.data.RelationMentions;
import edu.jhu.nlp.data.Span;
import edu.jhu.nlp.data.conll.SrlGraph;
import edu.jhu.nlp.features.TemplateLanguage.AT;
import edu.jhu.nlp.tag.StrictPosTagAnnotator.StrictPosTag;
import edu.jhu.pacaya.parse.cky.data.NaryTree;
import edu.jhu.pacaya.parse.dep.ParentsArray;
import edu.jhu.pacaya.util.collections.QLists;
import edu.jhu.prim.arrays.IntArrays;
import edu.jhu.prim.list.IntArrayList;
import edu.jhu.prim.set.IntHashSet;
import edu.jhu.prim.tuple.Pair;

/**
 * Simple representation of a single sentence with many annotations.
 * 
 * This representation only uses strings, without String objects or Alphabet objects.
 * 
 * @author mgormley
 * @author mmitchell
 */
public class AnnoSentence {
    
    private List<String> words;
    // 5-gram prefix if the word is longer than 5 characters.
    private List<String> prefixes;
    private List<String> lemmas;
    private List<String> posTags;
    private List<String> cposTags;
    private List<StrictPosTag> strictPosTags;
    private List<String> clusters;
    private IntArrayList embedIds;
    private List<List<String>> feats;
    private List<String> chunks;
    private List<String> neTags;
    private List<String> deprels;
    /**
     * Internal representation of a dependency parse: parents[i] gives the index
     * of the parent of the word at index i. The Wall node has index -1. If a
     * word has no parent, it has index -2 (e.g. if punctuation was not marked
     * with a head).
     */
    private int[] parents;
    private DepEdgeMask depEdgeMask;
    private IntHashSet knownPreds;
    private DepGraph srlGraph;
    /** Constituency parse. */
    private NaryTree naryTree;
    // The standard set of named entities.
    private NerMentions namedEntities;
    // Pairs of named entities to be considered for relation extraction.
    // This set could be all pairs, all ordered pairs, or some other definition.
    private List<Pair<NerMention,NerMention>> nePairs;
    // Labels for the pairs of named entities.
    private List<String> relLabels;
    // The standard set of relation mentions.
    private RelationMentions relations;
    
    /** The original object (e.g. CoNLL09Sentence) used to create this sentence. */
    private Object sourceSent;
    
    public AnnoSentence() {

    }

    /**
     * Fairly deep copy constructor. Everything is deeply copied except for the
     * source sentence and the SRL graph, the features, and the constituency parse.
     */
    @Deprecated
    public AnnoSentence getFairlyDeepCopy() {
        AnnoSentence newSent = new AnnoSentence();
        newSent.words = QLists.copyOf(this.words);
        newSent.prefixes = QLists.copyOf(this.prefixes);
        newSent.lemmas = QLists.copyOf(this.lemmas);
        newSent.posTags = QLists.copyOf(this.posTags);
        newSent.cposTags = QLists.copyOf(this.cposTags);
        newSent.strictPosTags = QLists.copyOf(this.strictPosTags);
        newSent.clusters = QLists.copyOf(this.clusters);
        newSent.embedIds = new IntArrayList(this.embedIds);
        newSent.chunks = QLists.copyOf(this.chunks);
        newSent.neTags = QLists.copyOf(this.neTags);
        newSent.deprels = QLists.copyOf(this.deprels);
        newSent.parents = IntArrays.copyOf(this.parents);
        newSent.depEdgeMask = (this.depEdgeMask == null) ? null : new DepEdgeMask(this.depEdgeMask);
        newSent.knownPreds = (this.knownPreds == null) ? null : new IntHashSet(this.knownPreds);
        newSent.namedEntities = new NerMentions(this.namedEntities);
        newSent.nePairs = QLists.copyOf(nePairs);
        newSent.relLabels = QLists.copyOf(relLabels);
        newSent.relations = new RelationMentions(this.relations);
        newSent.sourceSent = this.sourceSent;
        // TODO: this should be a deep copy.
        newSent.feats = QLists.copyOf(this.feats);
        // TODO: this should be a deep copy.
        newSent.srlGraph = this.srlGraph;
        // TODO: this should be a deep copy.
        newSent.naryTree = this.naryTree;
        return newSent;
    }
    
    public AnnoSentence getShallowCopy() {
        AnnoSentence newSent = new AnnoSentence();
        for (AT at : AT.values()) {
        	copyShallow(this, newSent, at);
        }
        return newSent;
    }

    public static void copyShallow(AnnoSentence src, AnnoSentence dest, AT at) {
        switch (at) {
        case WORD: dest.words = src.words; break;
        case PREFIX: dest.prefixes = src.prefixes; break;
        case LEMMA: dest.lemmas = src.lemmas; break;
        case POS: dest.posTags = src.posTags; break;
        case CPOS: dest.cposTags = src.cposTags; break;
        case STRICT_POS: dest.strictPosTags = src.strictPosTags; break;
        case BROWN: dest.clusters = src.clusters; break;
        case EMBED_IDX: dest.embedIds = src.embedIds; break;
        case MORPHO: dest.feats = src.feats; break;
        case CHUNKS: dest.chunks = src.chunks; break;
        case NE_TAGS: dest.neTags = src.neTags; break;
        case DEP_TREE: dest.parents = src.parents; break;
        case DEPREL: dest.deprels = src.deprels; break;
        case DEP_EDGE_MASK: dest.depEdgeMask = src.depEdgeMask; break;
        case SRL_PRED_IDX: dest.knownPreds = src.knownPreds; break;
        case SRL: dest.srlGraph = src.srlGraph; break;
        case NARY_TREE: dest.naryTree = src.naryTree; break;
        case NER: dest.namedEntities = src.namedEntities; break;
        case NE_PAIRS: dest.nePairs = src.nePairs; break;
        case REL_LABELS: dest.relLabels = src.relLabels; break;
        case RELATIONS: dest.relations = src.relations; break;
        default: throw new RuntimeException("not implemented for " + at);
        }
    }

    public void removeAts(Collection<AT> removeAts) {
        for (AT at : removeAts) {
            removeAt(at);
        }
    }

    public void removeAt(AT at) {
        switch (at) {
        case WORD: this.words = null; break;
        case PREFIX: this.prefixes = null; break;
        case LEMMA: this.lemmas = null; break;
        case POS: this.posTags = null; break;
        case CPOS: this.cposTags = null; break;
        case STRICT_POS: this.strictPosTags = null; break;
        case BROWN: this.clusters = null; break;
        case EMBED_IDX: this.embedIds = null; break;
        case MORPHO: this.feats = null; break;
        case CHUNKS: this.chunks = null; break;
        case NE_TAGS: this.neTags = null; break;
        case DEP_TREE: this.parents = null; break; // TODO: Should DEP_TREE also remove the labels? Not clear.
        case DEPREL: this.deprels = null; break;
        case DEP_EDGE_MASK: this.depEdgeMask = null; break;
        case SRL_PRED_IDX: this.knownPreds = null; break;
        case SRL: this.srlGraph = null; break;
        case NARY_TREE: this.naryTree = null; break;
        case NER: this.namedEntities = null; break;
        case NE_PAIRS: this.nePairs = null; break;
        case REL_LABELS: this.relLabels = null; break;
        case RELATIONS: this.relations = null; break;
        default: throw new RuntimeException("not implemented for " + at);
        }
    }
    
    public boolean hasAt(AT at) {
        switch (at) {
        case WORD: return this.words != null;
        case PREFIX: return this.prefixes != null;
        case LEMMA: return this.lemmas != null;
        case POS: return this.posTags != null;
        case CPOS: return this.cposTags != null;
        case STRICT_POS: return this.strictPosTags != null;
        case BROWN: return this.clusters != null;
        case EMBED_IDX: return this.embedIds != null;
        case MORPHO: return this.feats != null;
        case CHUNKS: return this.chunks != null;
        case NE_TAGS: return this.neTags != null;
        case DEP_TREE: return this.parents != null;
        case DEPREL: return this.deprels != null;
        case DEP_EDGE_MASK: return this.depEdgeMask != null;
        case SRL_PRED_IDX: return this.knownPreds != null;
        case SRL: return this.srlGraph != null;
        case NARY_TREE: return this.naryTree != null;
        case NER: return this.namedEntities != null;
        case NE_PAIRS: return this.nePairs != null;
        case REL_LABELS: return this.relLabels != null;
        case RELATIONS: return this.relations != null;        
        default: throw new RuntimeException("not implemented for " + at);
        }
    }
    
    public void intern() {
        QLists.intern(words);
        QLists.intern(prefixes);
        QLists.intern(lemmas);
        QLists.intern(posTags);
        QLists.intern(cposTags);
        // Not needed since these are enums. Lists.intern(strictPosTags);
        QLists.intern(clusters);
        if (feats != null) {
            for (int i=0; i<feats.size(); i++) {
                QLists.intern(feats.get(i));
            }
        }
        QLists.intern(chunks);
        QLists.intern(neTags);
        QLists.intern(deprels);        
        if (naryTree != null) {
            naryTree.intern();
        }
        if (namedEntities != null) {
            namedEntities.intern();
        }
        // TODO: Lists.intern(nePairs);
        QLists.intern(relLabels);
        if (relations != null) {
            relations.intern();
        }
        // TODO: this.srlGraph.intern();
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        appendIfNotNull(sb, "words", words);
        appendIfNotNull(sb, "prefixes", prefixes);
        appendIfNotNull(sb, "lemmas", lemmas);
        appendIfNotNull(sb, "posTags", posTags);
        appendIfNotNull(sb, "cposTags", cposTags);
        appendIfNotNull(sb, "strictPosTags", strictPosTags);
        appendIfNotNull(sb, "clusters", clusters);
        appendIfNotNull(sb, "embedIds", embedIds);
        appendIfNotNull(sb, "feats", feats);
        appendIfNotNull(sb, "chunks", chunks);
        appendIfNotNull(sb, "neTags", neTags);
        if (parents != null) {
            sb.append("parents=");
            sb.append(Arrays.toString(parents));
            sb.append(",\n");
        }
        appendIfNotNull(sb, "deprels", deprels);
        appendIfNotNull(sb, "depEdgeMask", depEdgeMask);
        appendIfNotNull(sb, "srlGraph", srlGraph);
        appendIfNotNull(sb, "knownPreds", knownPreds);
        appendIfNotNull(sb, "naryTree", naryTree);
        appendIfNotNull(sb, "namedEntities", namedEntities);
        if (namedEntities != null) { appendIfNotNull(sb, "namedEntities (context)", namedEntities.toString(words)); }
        appendIfNotNull(sb, "nePairs", nePairs);
        appendIfNotNull(sb, "relLabels", relLabels);
        appendIfNotNull(sb, "relations", relations);
        if (relations != null) { appendIfNotNull(sb, "relations (context)", relations.toString(words)); }
        appendIfNotNull(sb, "sourceSent", sourceSent);
        sb.append("]");
        return sb.toString();
    }

    private void appendIfNotNull(StringBuilder sb, String name, Object l) {
        if (l != null) {
            sb.append(name);
            sb.append("=");
            sb.append(l);
            sb.append(",\n");
        }
    }
    
    /* -------------------------------- Interesting getters / setters ---------------------------- */
    
    /** Gets the i'th word as a String. */
    public String getWord(int i) {
        return words.get(i);
    }
    
    /** Gets the i'th prefix of 5 characters as a String. */
    public String getPrefix(int i) {
        return prefixes.get(i);
    }

    /** Gets the i'th POS tag as a String. */
    public String getPosTag(int i) {
        return posTags.get(i);
    }

    /** Gets the i'th Coarse POS tag as a String. */
    public String getCposTag(int i) {
        return cposTags.get(i);
    }
    
    /** Gets the i'th Strict POS tag. */
    public StrictPosTag getStrictPosTag(int i) {
        return strictPosTags.get(i);
    }

    /** Gets the i'th Distributional Similarity Cluster ID as a String. */
    public String getCluster(int i) {
        return clusters.get(i);
    }
    
    public int getEmbedId(int i) {
        return embedIds.get(i);
    }
    
    /** Gets the i'th lemma as a String. */
    public String getLemma(int i) {
        return lemmas.get(i);
    }

    /** Gets the i'th chunk as a String. */
    public String getChunk(int i) {
        return chunks.get(i);
    }
    
    /** Gets the i'th chunk as a String. */
    public String getNeTag(int i) {
        return neTags.get(i);
    }
    
    /** Gets the index of the parent of the i'th word. */
    public int getParent(int i) {
        return parents[i];
    }

    /** Returns whether the corresponding dependency arc should be pruned. */
    public boolean isDepEdgePruned(int parent, int child) {
        return depEdgeMask.isPruned(parent, child);
    }
    
    /** Gets the features (e.g. morphological features) of the i'th word. */
    public List<String> getFeats(int i) {
        return feats.get(i);
    }

    /** Gets the dependency relation label for the arc from the i'th word to its parent. */
    public String getDeprel(int i) {
        // TODO: Decide whether we should always return null for these sorts of get calls.
        if (deprels == null) { return null; }
        return deprels.get(i);
    }
        
    /**
     * Gets a list of words corresponding to a token span.
     */
    public List<String> getWords(Span span) {
        return getSpan(words, span);
    }
    
    /**
     * Gets a list of words corresponding to a token span.
     */
    public List<String> getPrefixes(Span span) {
        return getSpan(prefixes, span);
    }

    /**
     * Gets a list of parent indices corresponding to a token span.
     */
    public List<Integer> getParents(Span span) {
        return getSpan(parents, span);
    }
    
    /**
     * Gets a list of POS tags corresponding to a token span.
     */
    public List<String> getPosTags(Span span) {
        return getSpan(posTags, span);
    }
    
    /**
     * Gets a list of coarse POS tags corresponding to a token span.
     */
    public List<String> getCposTags(Span span) {
        return getSpan(cposTags, span);
    }
    
    /**
     * Gets a list of strict POS tags corresponding to a token span.
     */
    public List<StrictPosTag> getStrictPosTags(Span span) {
        return getSpan(strictPosTags, span);
    }
    
    /**
     * Gets a list of Distributional Similarity Cluster IDs corresponding to a token span.
     */
    public List<String> getClusters(Span span) {
        return getSpan(clusters, span);
    }

    /**
     * Gets a list of lemmas corresponding to a token span.
     */
    public List<String> getLemmas(Span span) {
        return getSpan(lemmas, span);
    }
    
    // TODO: Consider moving this to LabelSequence.
    private static <T> List<T> getSpan(List<T> seq, Span span) {
        return seq.subList(span.start(), span.end());
    }

    private static List<Integer> getSpan(int[] parents, Span span) {
        assert (span != null);
        List<Integer> list = new ArrayList<Integer>();
        for (int i = span.start(); i < span.end(); i++) {
            list.add(i);
        }
        return list;
    }
    
    /**
     * Gets the shortest dependency path between two tokens.
     * 
     * <p>
     * For the tree: x0 <-- x1 --> x2, represented by parents=[1, -1, 1] the
     * dependency path from x0 to x2 would be a list [(0, UP), (1, DOWN)]
     * </p>
     * 
     * <p>
     * See DepTreeTest for examples.
     * </p>
     * 
     * @param start The position of the start token.
     * @param end The position of the end token.
     * @return The path as a list of pairs containing the word positions and the
     *         direction of the edge, inclusive of the start position and
     *         exclusive of the end.
     */
    public List<Pair<Integer, ParentsArray.Dir>> getDependencyPath(int start, int end) {
        return ParentsArray.getDependencyPath(start, end, parents);
    }
    
    public int size() {
        return words.size();
    }
    
    public boolean isKnownPred(int i) {
        return knownPreds.contains(i);
    }
    
    public void setKnownPredsFromSrlGraph() {
        if (srlGraph == null) {
            throw new IllegalStateException("This can only be called if srlGraph is non-null.");
        }
        knownPreds = new IntHashSet();
        for (int p=0; p<srlGraph.size(); p++) {
            if (srlGraph.get(-1, p) != null) {
                knownPreds.add(p);
            }
        }
    }
    
    /* ----------- Getters/Setters for internal storage ------------ */
        
    public List<String> getWords() {
        return words;
    }

    public void setWords(List<String> words) {
        this.words = words;
    }
    
    public List<String> getPrefixes() {
        return prefixes;
    }

    public void setPrefixes(List<String> prefixes) {
        this.prefixes = prefixes;
    }

    public List<String> getLemmas() {
        return lemmas;
    }

    public void setLemmas(List<String> lemmas) {
        this.lemmas = lemmas;
    }

    public List<String> getPosTags() {
        return posTags;
    }

    public void setPosTags(List<String> posTags) {
        this.posTags = posTags;
    }
    
    public List<String> getCposTags() {
        return cposTags;
    }

    public void setCposTags(List<String> cposTags) {
        this.cposTags = cposTags;
    }
        
    public List<StrictPosTag> getStrictPosTags() {
        return strictPosTags;
    }

    public void setStrictPosTags(List<StrictPosTag> strictPosTags) {
        this.strictPosTags = strictPosTags;
    }

    public List<String> getClusters() {
        return clusters;
    }

    public void setClusters(List<String> clusters) {
        this.clusters = clusters;
    }
        
    public IntArrayList getEmbedIds() {
        return embedIds;
    }

    public void setEmbedIds(IntArrayList embedIds) {
        this.embedIds = embedIds;
    }

    public List<String> getChunks() {
        return chunks;
    }

    public void setChunks(List<String> chunks) {
        this.chunks = chunks;
    }
    
    public List<String> getNeTags() {
        return neTags;
    }

    public void setNeTags(List<String> neTags) {
        this.neTags = neTags;
    }

    public int[] getParents() {
        return parents;
    }

    public void setParents(int[] parents) {
        this.parents = parents;
    }

    public DepEdgeMask getDepEdgeMask() {
        return depEdgeMask;
    }

    public void setDepEdgeMask(DepEdgeMask depEdgeMask) {
        this.depEdgeMask = depEdgeMask;
    }

    public IntHashSet getKnownPreds() {
        return knownPreds;
    }
    
    public void setKnownPreds(IntHashSet knownPreds) {
        this.knownPreds = knownPreds;
    }

    public DepGraph getSrlGraph() {
        return srlGraph;
    }

    /** Constructs a new list containing the predicate senses. */
    public List<String> getPredSenses() {
        if (srlGraph == null) {
            return null;
        }
        ArrayList<String> senses = new ArrayList<>(srlGraph.size());
        for (int p=0; p<srlGraph.size(); p++) {
            senses.add(srlGraph.get(-1, p));
        }
        return senses;
    }
    
    public void setSrlGraph(DepGraph srlGraph) {
        this.srlGraph = srlGraph;
    }
    
    public List<String> getDeprels() {
        return deprels;
    }

    public void setDeprels(List<String> deprels) {
        this.deprels = deprels;
    }

    public List<List<String>> getFeats() {
        return feats;
    }

    public void setFeats(List<List<String>> feats) {
        this.feats = feats;
    }
    
    public NaryTree getNaryTree() {
        return naryTree;
    }

    public void setNaryTree(NaryTree naryTree) {
        this.naryTree = naryTree;
    }
    
    /** Gets the original object (e.g. CoNLL09Sentence) used to create this sentence. */
    public Object getSourceSent() {
        return sourceSent;
    }
    
    /** Sets the original object (e.g. CoNLL09Sentence) used to create this sentence. */
    public void setSourceSent(Object sourceSent) {
        this.sourceSent = sourceSent;
    }

    public NerMentions getNamedEntities() {
        return namedEntities;
    }
    
    public void setNamedEntities(NerMentions namedEntities) {
        this.namedEntities = namedEntities;
    }
    
    public List<Pair<NerMention, NerMention>> getNePairs() {
		return nePairs;
	}

	public void setNePairs(List<Pair<NerMention, NerMention>> nePairs) {
		this.nePairs = nePairs;
	}

	public List<String> getRelLabels() {
        return relLabels;
    }

    public void setRelLabels(List<String> relLabels) {
        this.relLabels = relLabels;
    }

    public RelationMentions getRelations() {
        return relations;
    }

    public void setRelations(RelationMentions relations) {
        this.relations = relations;
    }

    public List<String> getLowerCaseWords() {
        if (words == null) { return null; }
        return words.stream()
                .map(x -> x.toLowerCase())
                .collect(Collectors.toList());
    }
    
}
