package edu.jhu.nlp.depparse;

import java.util.Collections;
import java.util.Set;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.Trainable;
import edu.jhu.nlp.data.DepEdgeMask;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.data.simple.AnnoSentenceCollection;
import edu.jhu.nlp.features.TemplateLanguage.AT;

public class GoldDepParseUnpruner implements Trainable {

    private static final Logger log = LoggerFactory.getLogger(GoldDepParseUnpruner.class);
    private static final long serialVersionUID = 1L;

    @Override
    public void annotate(AnnoSentenceCollection sents) {
        // Do nothing. This annotator is only used to manipulate training parses.
    }

    @Override
    public void train(AnnoSentenceCollection trainInput, AnnoSentenceCollection trainGold,
            AnnoSentenceCollection devInput, AnnoSentenceCollection devGold) {
        log.info("Ensuring that the gold trees in the training data are not pruned.");
        unprune(trainInput, trainGold);
    }

    protected void unprune(AnnoSentenceCollection trainInput, AnnoSentenceCollection trainGold) {
        for (int i=0; i<trainGold.size(); i++) {
            AnnoSentence gSent = trainGold.get(i);
            AnnoSentence iSent = trainInput.get(i);
            if (gSent.getParents() != null && iSent.getDepEdgeMask() != null) {
                int[] gParents = gSent.getParents();
                DepEdgeMask iMask = iSent.getDepEdgeMask();
                for (int c=0; c<gParents.length; c++) {
                    int p = gParents[c];
                    iMask.setIsPruned(p, c, false);
                }
            }
        }
    }
    
    @Override
    public Set<AT> getAnnoTypes() {
        return Collections.emptySet();
    }

}
