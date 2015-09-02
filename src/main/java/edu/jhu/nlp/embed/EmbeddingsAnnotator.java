package edu.jhu.nlp.embed;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.AbstractParallelAnnotator;
import edu.jhu.nlp.Annotator;
import edu.jhu.nlp.data.NerMention;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.data.simple.AnnoSentenceCollection;
import edu.jhu.nlp.embed.Embeddings.Scaling;
import edu.jhu.nlp.features.TemplateLanguage.AT;
import edu.jhu.pacaya.gm.feat.ObsFeatureConjoiner;
import edu.jhu.pacaya.util.Prm;
import edu.jhu.pacaya.util.collections.QSets;
import edu.jhu.prim.list.IntArrayList;
import edu.jhu.prim.set.IntHashSet;

public class EmbeddingsAnnotator extends AbstractParallelAnnotator implements Annotator {

    public static class EmbeddingsAnnotatorPrm extends Prm {
        private static final long serialVersionUID = 1L;
        // Path to word embeddings text file.
        public File embeddingsFile = null;
        // Method for normalization of the embeddings.
        public Scaling embNorm = Scaling.L1_NORM;
        // Amount to scale embeddings after normalization.
        public double embScalar = 15.0;
        // Whether to append the "-ne" suffix for entity specific embeddings.
        public boolean entitySpecificEmbeddings = false;
    }
    
    private static final long serialVersionUID = 1L;
    private static final Logger log = LoggerFactory.getLogger(Embeddings.class);
    private EmbeddingsAnnotatorPrm prm;
    private Embeddings embeddings;
    // Internal counting for miss rate.
    private AtomicInteger numLookups = new AtomicInteger(0);
    private AtomicInteger numMisses = new AtomicInteger(0);
    
    public EmbeddingsAnnotator(EmbeddingsAnnotatorPrm prm, Set<String> words) {
        this.prm = prm;
        this.embeddings = new Embeddings(prm.embeddingsFile, words);
        embeddings.normPerWord(prm.embNorm);
        embeddings.scaleAll(prm.embScalar);
        if (prm.entitySpecificEmbeddings) {
            log.info("Entity-specific embeddings enabled");
        }
    }
    
    public EmbeddingsAnnotator(EmbeddingsAnnotatorPrm prm, Embeddings embeddings) {
        if (prm.embeddingsFile != null) {
            throw new IllegalArgumentException("Embeddings already provided, so embeddingsFile shouldn't be set.");
        }
        this.prm = prm;
        this.embeddings = embeddings;
    }

    @Override
    public void annotate(AnnoSentenceCollection sents) {
        super.annotate(sents);
        log.info("Embeddings hit rate: " + getHitRate());                        
    }
    
    public void annotate(AnnoSentence sent) {
        IntArrayList embeds = new IntArrayList(sent.size());
        List<String> words = getWords(sent);
        for (int i=0; i<sent.size(); i++) {
            String word = words.get(i);
            embeds.add(embeddings.findEmbedding(word));
            if (embeds.get(i) == -1) {
                log.trace("Word not found: {}", word);
                numMisses.incrementAndGet();
            }
            numLookups.incrementAndGet();
        }
        sent.setEmbedIds(embeds);
    }

    protected List<String> getWords(AnnoSentence sent) {
        if (prm.entitySpecificEmbeddings) {
            return getEntitySpecificWords(sent);
        } else {
            return sent.getWords();
        }
    }

    public static List<String> getEntitySpecificWords(AnnoSentence sent) {
        final String suffix = "-ne";
        if (sent.getNamedEntities().size() > 2) {
            throw new IllegalArgumentException("Only able to annotate sentences containing at most 2 entity mentions.");
        }
        
        // Get the set of all named entity mention heads.
        IntHashSet neHeads = new IntHashSet();
        for (NerMention ne : sent.getNamedEntities()) {
            neHeads.add(ne.getHead());
        }
        // Append "-ne" to each entity mention WORD. 
        // This will allow us to use an entity mention specific embedding.
        List<String> words = new ArrayList<>();
        for (int i=0; i<sent.size(); i++) {
            if (neHeads.contains(i)) {
                words.add(sent.getWord(i) + suffix);
            } else {
                words.add(sent.getWord(i));
            }
        }
        return words;
    }

    public double getHitRate() {
        return (double) (numLookups.get() - numMisses.get()) / numLookups.get();
    }
    
    @Override
    public Set<AT> getAnnoTypes() {
        return QSets.getSet(AT.EMBED_IDX);
    }
    
    public Embeddings getEmbeddings() {
        return embeddings;
    }

}
