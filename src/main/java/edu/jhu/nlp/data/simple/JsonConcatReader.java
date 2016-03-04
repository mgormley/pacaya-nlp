package edu.jhu.nlp.data.simple;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map.Entry;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonStreamParser;

import edu.jhu.nlp.data.semeval.SemEval2010Reader;
import edu.jhu.nlp.tag.StrictPosTagAnnotator.StrictPosTag;
import edu.jhu.pacaya.parse.cky.data.NaryTree;

/**
 * Reads annotated sentences from a Concatenated JSON file.
 * 
 * See a description of concatenated JSON here:
 * https://en.wikipedia.org/wiki/JSON_Streaming#Concatenated_JSON.
 * 
 * @author mgormley
 */
public class JsonConcatReader implements CloseableIterable<AnnoSentence>, Iterator<AnnoSentence> {
    
    private static final Logger log = LoggerFactory.getLogger(SemEval2010Reader.class);
    private AnnoSentence sentence;
    private Reader reader;
    private JsonStreamParser parser;

    public JsonConcatReader(File file) throws IOException {        
        this(new FileInputStream(file));
    }

    public JsonConcatReader(InputStream inputStream) throws UnsupportedEncodingException {
        this(new BufferedReader(new InputStreamReader(inputStream, "UTF-8")));
    }

    private JsonConcatReader(Reader reader) {
        this.reader = reader;
        this.parser = new JsonStreamParser(reader);
        next();
    }
    
    public static AnnoSentence readSentence(JsonStreamParser reader) throws IOException {
        if(!reader.hasNext()) {
            return null;
        }
        JsonObject jo = reader.next().getAsJsonObject();
        AnnoSentence sent = new AnnoSentence();
        for (Entry<String,JsonElement> entry : jo.entrySet()) {
            String key = entry.getKey();
            JsonElement val = entry.getValue();
            // Process each key in this sentence.
            if (key.equals("words")) {
                sent.setWords(getListOfStrings(key, val));
            } else if (key.equals("prefixes")) {
                sent.setPrefixes(getListOfStrings(key, val));
            } else if (key.equals("lemmas")) {
                sent.setLemmas(getListOfStrings(key, val));
            } else if (key.equals("posTags")) {
                sent.setPosTags(getListOfStrings(key, val));
            } else if (key.equals("cposTags")) {
                sent.setCposTags(getListOfStrings(key, val));
            } else if (key.equals("strictPosTags")) {
                List<String> tags = getListOfStrings(key, val);
                sent.setStrictPosTags(tags.stream().map(x -> StrictPosTag.valueOf(x)).collect(Collectors.toList()));
            } else if (key.equals("clusters")) {
                sent.setClusters(getListOfStrings(key, val));
            } else if (key.equals("chunks")) {
                sent.setChunks(getListOfStrings(key, val));
            } else if (key.equals("neTags")) {
                sent.setNeTags(getListOfStrings(key, val));
            } else if (key.equals("parents")) {
                sent.setParents(getIntArray(key, val));
            } else if (key.equals("deprels")) {
                sent.setDeprels(getListOfStrings(key, val));
            } else if (key.equals("naryTree")) {
                sent.setNaryTree(NaryTree.fromTreeInPtbFormat(val.getAsString()));
            } else if (key.equals("nePairs")) {
                sent.setNePairs(SimpleTextWriter.nePairsFromJson(val.getAsString()));
            } else if (key.equals("relLabels")) {
                sent.setRelLabels(getListOfStrings(key, val));
            } else{
                // Not supported: 
                // - embedIds (IntArrayList)
                // - feats (List<List<String>>)
                // - depEdgeMask (DepEdgeMask)
                // - srlGraph (SrlGraph)
                // - knownPreds (IntHashSet)
                // - namedEntities (NerMentions)
                // - namedEntitiesInContext
                // - relations (RelationMentions)
                // - relationsInContext
                throw new RuntimeException("Unsupported key:" + key);
            }
        }
        
        return sent;
    }

    private static int[] getIntArray(String key, JsonElement val) {
        JsonArray a = val.getAsJsonArray();
        int[] ints = new int[a.size()];
        for (int i=0; i<a.size(); i++) {
            ints[i] = a.get(i).getAsInt();
        }
        return ints;
    }

    private static List<String> getListOfStrings(String key, JsonElement val) {
        JsonArray a = val.getAsJsonArray();
        List<String> strs = new ArrayList<>(a.size());
        for (int i=0; i<a.size(); i++) {
            strs.add(a.get(i).getAsString());    
        }
        return strs;
    }

    @Override
    public boolean hasNext() {
        return sentence != null;
    }

    @Override
    public AnnoSentence next() {
        try {
            AnnoSentence curSent = sentence;
            sentence = readSentence(parser);
            if (curSent != null) {
                curSent.intern();
            }
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
    public Iterator<AnnoSentence> iterator() {
        return this;
    }

    public void close() throws IOException {
        reader.close();
    }

}
