package edu.jhu.nlp.data.simple;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Collectors;

import javax.json.Json;
import javax.json.JsonArray;
import javax.json.JsonObject;
import javax.json.JsonReader;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonStreamParser;

import edu.jhu.nlp.data.semeval.SemEval2010Reader;
import edu.jhu.nlp.tag.StrictPosTagAnnotator.StrictPosTag;
import edu.jhu.pacaya.parse.cky.data.NaryTree;

public class JsonSentReader implements CloseableIterable<AnnoSentence>, Iterator<AnnoSentence> {
    
    private static final Logger log = LoggerFactory.getLogger(SemEval2010Reader.class);
    private AnnoSentence sentence;
    private BufferedReader reader;

    public JsonSentReader(File file) throws IOException {        
        this(new FileInputStream(file));
    }

    public JsonSentReader(InputStream inputStream) throws UnsupportedEncodingException {
        this(new BufferedReader(new InputStreamReader(inputStream, "UTF-8")));
    }

    private JsonSentReader(BufferedReader reader) {
        this.reader = reader;
        next();
    }
    
    public static AnnoSentence readSentence(BufferedReader reader) throws IOException {
        JsonReader r = Json.createReader(reader);
        JsonObject jo = r.readObject();
        AnnoSentence sent = new AnnoSentence();
        for (String key : jo.keySet()) {
            // Process each key in this sentence.
            if (key.equals("words")) {
                sent.setWords(getListOfStrings(jo, key));
            } else if (key.equals("prefixes")) {
                sent.setPrefixes(getListOfStrings(jo, key));
            } else if (key.equals("lemmas")) {
                sent.setLemmas(getListOfStrings(jo, key));
            } else if (key.equals("posTags")) {
                sent.setPosTags(getListOfStrings(jo, key));
            } else if (key.equals("cposTags")) {
                sent.setCposTags(getListOfStrings(jo, key));
            } else if (key.equals("strictPosTags")) {
                List<String> tags = getListOfStrings(jo, key);
                sent.setStrictPosTags(tags.stream().map(x -> StrictPosTag.valueOf(x)).collect(Collectors.toList()));
            } else if (key.equals("clusters")) {
                sent.setClusters(getListOfStrings(jo, key));
            } else if (key.equals("chunks")) {
                sent.setChunks(getListOfStrings(jo, key));
            } else if (key.equals("neTags")) {
                sent.setNeTags(getListOfStrings(jo, key));
            } else if (key.equals("parents")) {
                sent.setParents(getIntArray(jo, key));
            } else if (key.equals("deprels")) {
                sent.setDeprels(getListOfStrings(jo, key));
            } else if (key.equals("naryTree")) {
                sent.setNaryTree(NaryTree.fromTreeInPtbFormat(jo.getString(key)));
            } else if (key.equals("nePairs")) {
                sent.setNePairs(SimpleTextWriter.nePairsFromJson(jo.getString(key)));
            } else if (key.equals("relLabels")) {
                sent.setRelLabels(getListOfStrings(jo, key));
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

    private static int[] getIntArray(JsonObject jo, String key) {
        JsonArray a = jo.getJsonArray(key);
        int[] ints = new int[a.size()];
        for (int i=0; i<a.size(); i++) {
            ints[i] = a.getInt(i);
        }
        return ints;
    }

    private static List<String> getListOfStrings(JsonObject jo, String key) {
        JsonArray a = jo.getJsonArray(key);
        List<String> strs = new ArrayList<>(a.size());
        for (int i=0; i<a.size(); i++) {
            strs.add(a.getString(i));    
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
            sentence = readSentence(reader);
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
