package edu.jhu.nlp.data.conll;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.data.conll.SrlGraph.SrlArg;
import edu.jhu.nlp.data.conll.SrlGraph.SrlEdge;
import edu.jhu.nlp.data.conll.SrlGraph.SrlPred;
import edu.jhu.nlp.data.simple.AnnoSentence;

/**
 * One sentence from a CoNLL-2002 formatted file.
 * @author mgormley
 */
public class CoNLL02Sentence implements Iterable<CoNLL02Token> {

    private static Logger log = LoggerFactory.getLogger(CoNLL02Sentence.class);
    
    private ArrayList<CoNLL02Token> tokens;
    
    public CoNLL02Sentence(List<CoNLL02Token> tokens) {
        this.tokens = new ArrayList<CoNLL02Token>(tokens);
    }

    /** Deep copy constructor. */
    public CoNLL02Sentence(CoNLL02Sentence sent) {
        tokens = new ArrayList<CoNLL02Token>(sent.tokens.size());
        for (CoNLL02Token tok : sent) {
            tokens.add(new CoNLL02Token(tok));
        }
    }

    public static CoNLL02Sentence getInstanceFromTokenStrings(ArrayList<String> sentLines) {
        List<CoNLL02Token> tokens = new ArrayList<CoNLL02Token>();
        for (String line : sentLines) {
            tokens.add(new CoNLL02Token(line));
        }
        return new CoNLL02Sentence(tokens);
    }

    public CoNLL02Token get(int i) {
        return tokens.get(i);
    }

    public int size() {
        return tokens.size();
    }

    @Override
    public Iterator<CoNLL02Token> iterator() {
        return tokens.iterator();
    }
    
    public AnnoSentence toAnnoSentence() {
        AnnoSentence s = new AnnoSentence();
        s.setSourceSent(this);
        s.setWords(this.getWords());
        s.setPosTags(this.getPosTags());
        s.setNeTags(this.getNeTags());
        return s;
    }

    public List<String> getWords() {
        List<String> words = new ArrayList<String>(size());
        for (int i=0; i<size(); i++) {
            words.add(tokens.get(i).getWord());            
        }
        return words;
    }
    
    public List<String> getPosTags() {
        List<String> posTags = new ArrayList<String>(size());
        for (int i=0; i<size(); i++) {
            posTags.add(tokens.get(i).getPos());            
        }
        return posTags;
    }

    public List<String> getNeTags() {
        List<String> neTags = new ArrayList<String>(size());
        for (int i=0; i<size(); i++) {
            neTags.add(tokens.get(i).getNe());            
        }
        return neTags;
    }
    

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + ((tokens == null) ? 0 : tokens.hashCode());
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (getClass() != obj.getClass())
            return false;
        CoNLL02Sentence other = (CoNLL02Sentence) obj;
        if (tokens == null) {
            if (other.tokens != null)
                return false;
        } else if (!tokens.equals(other.tokens))
            return false;
        return true;
    }    
 
    public String toString() {
        try {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            CoNLL02Writer writer = new CoNLL02Writer(new OutputStreamWriter(baos));
            writer.write(this);
            writer.close();
            return baos.toString("UTF-8");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
