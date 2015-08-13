package edu.jhu.nlp.relations;

import java.util.List;
import java.util.Set;

import edu.jhu.nlp.AbstractParallelAnnotator;
import edu.jhu.nlp.Annotator;
import edu.jhu.nlp.data.NerMention;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.features.TemplateLanguage.AT;
import edu.jhu.pacaya.util.collections.QSets;
import edu.jhu.prim.iter.IntIter;
import edu.jhu.prim.set.IntHashSet;

/**
 * Adds the suffix "-ne" to entity mention head words. This is only for use with the SemEval-2010
 * Task 8 dataset and the accompanying entity-specific embeddings. Notably, this approach assumes that
 * there is at most one pair of named entities per sentence.
 * 
 * @author mgormley
 */
public class EntityMentionMunger extends AbstractParallelAnnotator implements Annotator {

    private static final long serialVersionUID = 1L;

    private String suffix;
    
    public EntityMentionMunger() {
        this("-ne");
    }
    
    public EntityMentionMunger(String suffix) {
        this.suffix = suffix;
    }
    
    @Override
    public void annotate(AnnoSentence sent) {
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
        IntIter iter = neHeads.iterator();
        List<String> words = sent.getWords();
        while (iter.hasNext()) {
            int i = iter.next();
            words.set(i, sent.getWord(i) + suffix);
        }
        sent.setWords(words);
    }
    
    @Override
    public Set<AT> getAnnoTypes() {
        return QSets.getSet(AT.WORD);
    }

}
