package edu.jhu.nlp.data.conll;

import edu.jhu.nlp.data.DepTree;
import edu.jhu.pacaya.nlp.data.Sentence;
import edu.jhu.pacaya.util.Alphabet;

/**
 * Dependency tree that carries the original CoNLL-X sentence as metadata.
 * 
 * @author mgormley
 *
 */
public class CoNLLXDepTree extends DepTree {

    private CoNLLXSentence sent;
    
    public CoNLLXDepTree(CoNLLXSentence sent, Alphabet<String> alphabet) {
        // TODO: filter out punctuation.
        super(new CXWrappedSentence(sent, alphabet), sent.getParentsFromHead(), false);
        this.sent = sent;
    }

    public CoNLLXSentence getCoNLLXSentence() {
        return sent;
    }

    // This class adds an alternative constructor to Sentence.
    private static class CXWrappedSentence extends Sentence {

        private static final long serialVersionUID = 1L;

        public CXWrappedSentence(CoNLLXSentence sent, Alphabet<String> alphabet) {
            super(alphabet);
            for (CoNLLXToken token : sent) {
                // TODO: Here we just add the tags.
                add(token.getPosTag());
            }
        }
        
    }

}
