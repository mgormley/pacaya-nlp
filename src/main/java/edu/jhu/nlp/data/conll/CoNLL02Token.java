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
public class CoNLL02Token {

    private static Logger log = LoggerFactory.getLogger(CoNLL02Token.class);

    private static final Pattern whitespace = Pattern.compile("\\s+");
    
    // Field number:    Field name:     Description:
    /** 1    Word  */
    private String word;
    /** 2    Part-of-speech tag */
    private String pos;
    /** 3    Named Entity Tag*/
    private String ne;
    
    public CoNLL02Token(String line) {
        String[] splits = whitespace.split(line);
        if (splits.length < 3) {
            throw new IllegalStateException("Line is incomplete: " + line);
        }
        word = splits[0].trim();
        pos = splits[1].trim();
        ne = splits[2].trim();
    }
    
    public CoNLL02Token(String word, String pos, String ne) {
        super();
        this.word = word;
        this.pos = pos;
        this.ne = ne;
    }

    /** Deep copy constructor */
    public CoNLL02Token(CoNLL02Token other) {
        this.word = other.word;
        this.pos = other.pos;
        this.ne = other.ne;
    }

    public void write(Writer writer) throws IOException {
        writer.write(String.format("%s %s %s", word, pos, ne));
    }
    
    /* ------------- Auto-generated Methods --------------- */
    
    @Override
    public String toString() {
        return "CoNLL02Token [word=" + word + ", pos=" + pos + ", ne=" + ne + "]";
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
        CoNLL02Token other = (CoNLL02Token) obj;
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
