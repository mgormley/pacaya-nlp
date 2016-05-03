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
public class CoNLL03Sentence implements Iterable<CoNLL03Token> {

    private static Logger log = LoggerFactory.getLogger(CoNLL03Sentence.class);
    
    private ArrayList<CoNLL03Token> tokens;
    
    public CoNLL03Sentence(List<CoNLL03Token> tokens) {
        this.tokens = new ArrayList<CoNLL03Token>(tokens);
    }

    /** Deep copy constructor. */
    public CoNLL03Sentence(CoNLL03Sentence sent) {
        tokens = new ArrayList<CoNLL03Token>(sent.tokens.size());
        for (CoNLL03Token tok : sent) {
            tokens.add(new CoNLL03Token(tok));
        }
    }

    public static CoNLL03Sentence getInstanceFromTokenStrings(ArrayList<String> sentLines) {
        List<CoNLL03Token> tokens = new ArrayList<CoNLL03Token>();
        for (String line : sentLines) {
            tokens.add(new CoNLL03Token(line));
        }
        return new CoNLL03Sentence(tokens);
    }

    public CoNLL03Token get(int i) {
        return tokens.get(i);
    }

    public int size() {
        return tokens.size();
    }

    @Override
    public Iterator<CoNLL03Token> iterator() {
        return tokens.iterator();
    }
    
    public AnnoSentence toAnnoSentence() {
        AnnoSentence s = new AnnoSentence();
        s.setSourceSent(this);
        s.setWords(this.getWords());
        s.setPosTags(this.getPosTags());
        s.setChunks(this.getChunkTags());
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

    public List<String> getChunkTags() {
        List<String> chunkTags = new ArrayList<String>(size());
        for (int i=0; i<size(); i++) {
            chunkTags.add(tokens.get(i).getChunk());            
        }
        return chunkTags;
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
        CoNLL03Sentence other = (CoNLL03Sentence) obj;
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
            CoNLL03Writer writer = new CoNLL03Writer(new OutputStreamWriter(baos));
            writer.write(this);
            writer.close();
            return baos.toString("UTF-8");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
