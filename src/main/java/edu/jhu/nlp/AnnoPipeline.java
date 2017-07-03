package edu.jhu.nlp;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.data.simple.AnnoSentenceCollection;
import edu.jhu.nlp.features.TemplateLanguage.AT;

/**
 * Pipeline of Annotators each of which is optionally trainable.
 * 
 * @author mgormley
 */
public class AnnoPipeline implements Trainable {

    private static final long serialVersionUID = 1L;
    private static final Logger log = LoggerFactory.getLogger(AnnoPipeline.class);
    private List<Annotator> pipeline = new ArrayList<Annotator>();
    
    public void add(Annotator anno) {
        pipeline.add(anno);
    }
    
    @Override
    public void train(AnnoSentenceCollection trainInput, AnnoSentenceCollection trainGold, 
            AnnoSentenceCollection devInput, AnnoSentenceCollection devGold) {
        throw new UnsupportedOperationException("train() is not supported. Use trainAndAnnotate() instead.");
    }
    
    @Override
    public void trainAndAnnotate(AnnoSentenceCollection trainInput, AnnoSentenceCollection trainGold, 
            AnnoSentenceCollection devInput, AnnoSentenceCollection devGold) {
        for (Annotator anno : pipeline) {
            if (anno instanceof Trainable) {
                // Train and annotate.
                ((Trainable) anno).trainAndAnnotate(trainInput, trainGold, devInput, devGold);
            } else {
                // Only annotate.
                anno.annotate(trainInput);
                if (devInput != null) {
                    anno.annotate(devInput);
                }
            }
        }
    }
    
    @Override
    public void annotate(AnnoSentenceCollection sents) {
        for (Annotator anno : pipeline) {
            anno.annotate(sents);
        }
    }
    
    @Override
    public Set<AT> getAnnoTypes() {
        HashSet<AT> ats = new HashSet<>();
        for (Annotator anno : pipeline) {
            ats.addAll(anno.getAnnoTypes());
        }
        return ats;
    }

}
