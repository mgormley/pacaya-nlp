package edu.jhu.hltcoe.data.conll;

import java.util.ArrayList;
import java.util.Iterator;

/**
 * One sentence from a CoNLL-2009 formatted file.
 */
public class CoNLL09Sentence implements Iterable<CoNLL09Token> {

    private ArrayList<CoNLL09Token> tokens;
    
    public CoNLL09Sentence(ArrayList<String> sentLines) {
        tokens = new ArrayList<CoNLL09Token>();
        for (String line : sentLines) {
            tokens.add(new CoNLL09Token(line));
        }
    }

//    public CoNLL09Sentence(Sentence sent, int[] heads) {
//        tokens = new ArrayList<CoNLL09Token>();
//        for (int i=0; i<sent.size(); i++) {
//            Label label = sent.get(i);
//            TaggedWord tw = (TaggedWord) label;
//            tokens.add(new CoNLL09Token(i+1, tw.getWord(), tw.getWord(), tw.getTag(), tw.getTag(), null, heads[i], "NO_LABEL", null, null));
//        }
//    }

    /** Deep copy constructor. */
    public CoNLL09Sentence(CoNLL09Sentence sent) {
        tokens = new ArrayList<CoNLL09Token>(sent.tokens.size());
        for (CoNLL09Token tok : sent) {
            tokens.add(new CoNLL09Token(tok));
        }
    }    
    
    public CoNLL09Token get(int i) {
        return tokens.get(i);
    }
    
    public int size() {
        return tokens.size();
    }
    
    @Override
    public Iterator<CoNLL09Token> iterator() {
        return tokens.iterator();
    }

    /**
     * Returns the head value for each token. The wall has index 0.
     * @return
     */
    public int[] getHeads() {
        int[] heads = new int[size()];
        for (int i=0; i<heads.length; i++) {
            heads[i] = tokens.get(i).getHead();
        }
        return heads;
    }
    
    /**
     * Returns my internal reprensentation of the parent index for each token. The wall has index -1.
     * @return
     */
    public int[] getParents() {
        int[] parents = new int[size()];
        for (int i=0; i<parents.length; i++) {
            parents[i] = tokens.get(i).getHead() - 1;
        }
        return parents;
    }

    public void setPheadsFromParents(int[] parents) {
        for (int i=0; i<parents.length; i++) {
            tokens.get(i).setPhead(parents[i] + 1);
        }
    }
}