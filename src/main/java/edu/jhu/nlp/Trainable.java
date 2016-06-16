package edu.jhu.nlp;

import edu.jhu.nlp.data.simple.AnnoSentenceCollection;

public interface Trainable extends Annotator {

    void train(AnnoSentenceCollection trainInput, AnnoSentenceCollection trainGold, 
            AnnoSentenceCollection devInput, AnnoSentenceCollection devGold);
    
}
