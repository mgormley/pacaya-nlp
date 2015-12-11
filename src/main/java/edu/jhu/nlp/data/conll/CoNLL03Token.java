package edu.jhu.nlp.data.conll;

import java.io.IOException;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Pattern;

import org.apache.commons.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * One token from a CoNLL 2002 formatted file. 
 * 
 * For details see: http://www.cnts.ua.ac.be/conll2002/ner/
 * 
 * @author mgormley
 * 
 */
public class CoNLL03Token {

    private static Logger log = LoggerFactory.getLogger(CoNLL03Token.class);

    private static final Pattern whitespace = Pattern.compile("\\s+");
    
    // Field number:    Field name:     Description:
    /** 1    Word  */
    private String word;
    /** 2    Part-of-speech tag */
    private String pos;
    /** 3    Chunk tag */
    private String chunk;
    /** 4    Named Entity Tag*/
    private String ne;
    
    public CoNLL03Token(String line) {
        String[] splits = whitespace.split(line);
        if (splits.length < 3) {
            throw new IllegalStateException("Line is incomplete: " + line);
        }
        word = splits[0].trim();
        pos = splits[1].trim();
        chunk = splits[2].trim();
        ne= splits[3].trim();
    }
    
    public CoNLL03Token(String word, String pos, String chunk, String ne) {
        super();
        this.word = word;
        this.pos = pos;
        this.chunk = chunk;
        this.ne = ne;
    }

    /** Deep copy constructor */
    public CoNLL03Token(CoNLL03Token other) {
        this.word = other.word;
        this.pos = other.pos;
        this.chunk = other.chunk;
        this.ne = other.ne;
    }

    public void write(Writer writer) throws IOException {
        writer.write(String.format("%s %s %s %s", word, pos, chunk, ne));
    }
    
    /* ------------- Auto-generated Methods --------------- */
    
    @Override
    public String toString() {
        return "CoNLL03Token [word=" + word + ", pos=" + pos + ", chunk=" + chunk + ", ne=" + ne + "]";
    }

    public String getWord() {
        return word;
    }

    public void setWord(String word) {
        this.word = word;
    }

    public String getPos() {
        return pos;
    }

    public void setPos(String pos) {
        this.pos = pos;
    }

    public String getChunk() {
        return chunk;
    }

    public void setChunk(String chunk) {
        this.chunk = chunk;
    }

    public String getNe() {
        return ne;
    }

    public void setNe(String ne) {
        this.ne = ne;
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + ((chunk == null) ? 0 : chunk.hashCode());
        result = prime * result + ((ne == null) ? 0 : ne.hashCode());
        result = prime * result + ((pos == null) ? 0 : pos.hashCode());
        result = prime * result + ((word == null) ? 0 : word.hashCode());
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
        CoNLL03Token other = (CoNLL03Token) obj;
        if (chunk == null) {
            if (other.chunk != null)
                return false;
        } else if (!chunk.equals(other.chunk))
            return false;
        if (ne == null) {
            if (other.ne != null)
                return false;
        } else if (!ne.equals(other.ne))
            return false;
        if (pos == null) {
            if (other.pos != null)
                return false;
        } else if (!pos.equals(other.pos))
            return false;
        if (word == null) {
            if (other.word != null)
                return false;
        } else if (!word.equals(other.word))
            return false;
        return true;
    }
        
}
