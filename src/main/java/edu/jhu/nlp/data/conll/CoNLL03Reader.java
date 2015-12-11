package edu.jhu.nlp.data.conll;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.regex.Pattern;

import edu.jhu.nlp.data.simple.CloseableIterable;

/**
 * Reads a single file in CoNLL-2002 format.
 * 
 * @author mgormley
 */
public class CoNLL03Reader implements CloseableIterable<CoNLL03Sentence>, Iterator<CoNLL03Sentence> {

    private CoNLL03Sentence sentence;
    private BufferedReader reader;

    public CoNLL03Reader(File file) throws IOException {
        this(new FileInputStream(file));
    }

    public CoNLL03Reader(InputStream inputStream) throws UnsupportedEncodingException {
        this(new BufferedReader(new InputStreamReader(inputStream, "iso-8859-1")));
    }
    
    public CoNLL03Reader(BufferedReader reader) {
        this.reader = reader;
        next();
    }

    public static CoNLL03Sentence readCoNLL03Sentence(BufferedReader reader) throws IOException {
        // The current token.
        String line;
        // The tokens for one sentence.
        ArrayList<String> tokens = new ArrayList<String>();

        while ((line = reader.readLine()) != null) {
            if (line.trim().equals("")) {
                // End of sentence marker.
                break;
            } else {
                // Regular token.
                tokens.add(line);
            }
        }
        if (tokens.size() > 0) {
            return CoNLL03Sentence.getInstanceFromTokenStrings(tokens);
        } else {
            return null;
        }
    }

    @Override
    public boolean hasNext() {
        return sentence != null;
    }

    @Override
    public CoNLL03Sentence next() {
        try {
            CoNLL03Sentence curSent = sentence;
            sentence = readCoNLL03Sentence(reader);
            return curSent;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void remove() {
        throw new RuntimeException("not implemented");
    }

    @Override
    public Iterator<CoNLL03Sentence> iterator() {
        return this;
    }

    public void close() throws IOException {
        reader.close();
    }

    public List<CoNLL03Sentence> readAll() {
        ArrayList<CoNLL03Sentence> sents = new ArrayList<CoNLL03Sentence>();
        for (CoNLL03Sentence sent : this) {
            sents.add(sent);
        }
        return sents;
    }

    public List<CoNLL03Sentence> readSents(int maxSents) {
        ArrayList<CoNLL03Sentence> sents = new ArrayList<CoNLL03Sentence>();
        for (CoNLL03Sentence sent : this) {
            if (sents.size() > maxSents) {
                break;
            }
            sents.add(sent);
        }
        return sents;
    }

}