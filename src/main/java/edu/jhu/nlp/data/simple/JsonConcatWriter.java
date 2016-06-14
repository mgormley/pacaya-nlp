package edu.jhu.nlp.data.simple;

import java.io.Closeable;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.StringReader;
import java.io.Writer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.json.Json;
import javax.json.JsonArray;
import javax.json.JsonArrayBuilder;
import javax.json.JsonObject;
import javax.json.JsonReader;
import javax.json.JsonValue;
import javax.json.stream.JsonGenerator;
import javax.json.stream.JsonGeneratorFactory;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.data.NerMention;
import edu.jhu.nlp.data.NerMentions;
import edu.jhu.nlp.data.Span;
import edu.jhu.prim.tuple.Pair;

/**
 * Writes annotated sentences to a Concatentated JSON file.  
 * 
 * See a description of concatenated JSON here:
 * https://en.wikipedia.org/wiki/JSON_Streaming#Concatenated_JSON.
 * 
 * @author mgormley
 */
public class JsonConcatWriter implements Closeable {

    private static final Logger log = LoggerFactory.getLogger(JsonConcatWriter.class);
    private JsonGenerator g;
    private Writer writer;    
    private int count;
    
    public JsonConcatWriter(File path) throws IOException {
        this(new FileOutputStream(path));
    }
    
    public JsonConcatWriter(OutputStream os) throws IOException {
        writer = new OutputStreamWriter(os, "UTF-8");
        this.count = 0;
    }
    
    public void write(AnnoSentenceCollection sents) throws IOException {
        for (AnnoSentence sent : sents) { this.write(sent); }
    }
    
    public void close() throws IOException {
        writer.close();
    }
        
    public int getCount() {
        return count;
    }
    
    private void forceNewline() throws IOException {
        g.flush();
        writer.write("\n");
    }

    public void write(AnnoSentence sent) throws IOException {
        Map<String, Object> properties = new HashMap<>(1);
        //properties.put(JsonGenerator.PRETTY_PRINTING, false);
        JsonGeneratorFactory gf = Json.createGeneratorFactory(properties);
        g = gf.createGenerator(writer);
        
        g.writeStartObject();
        appendIfNotNull("words", sent.getWords(), false);
        appendIfNotNull("prefixes", sent.getPrefixes());
        appendIfNotNull("lemmas", sent.getLemmas());
        appendIfNotNull("posTags", sent.getPosTags());
        appendIfNotNull("cposTags", sent.getCposTags());
        appendIfNotNull("strictPosTags", sent.getStrictPosTags());
        appendIfNotNull("clusters", sent.getClusters());
        appendIfNotNull("feats", sent.getFeats());
        appendIfNotNull("chunks", sent.getChunks());
        appendIfNotNull("neTags", sent.getNeTags());
        appendIfNotNull("parents", sent.getParents());
        appendIfNotNull("deprels", sent.getDeprels());
        appendIfNotNull("naryTree", sent.getNaryTree() == null ? null : sent.getNaryTree().getAsOneLineString());
        appendIfNotNull("namedEntities", sent.getNamedEntities() == null ? null : nesToJson(sent.getNamedEntities()).toString());
        appendIfNotNull("nePairs", nePairsToJson(sent.getNePairs()).toString());
        appendIfNotNull("relLabels", sent.getRelLabels());
        
        // Not included:
        //appendIfNotNull("embedIds", sent.getEmbedIds());
        //appendIfNotNull("depEdgeMask", sent.getDepEdgeMask());
        //appendIfNotNull("srlGraph", sent.getSrlGraph());
        //appendIfNotNull("knownPreds", sent.getKnownPreds());
        //if (sent.getNamedEntities() != null) { 
        // appendIfNotNull("namedEntities-InContext", sent.getNamedEntities().toString(sent.getWords())); 
        //}
        //appendIfNotNull("relations", sent.getRelations());
        //if (sent.getRelations() != null) {
        //   appendIfNotNull("relations-InContext", sent.getRelations().toString(sent.getWords())); 
        //}
        
        g.writeEnd();
        forceNewline();
        forceNewline();
        g.flush();
        count++;
    }

    private <T> void appendIfNotNull(String name, Object o) throws IOException {
        appendIfNotNull(name, o, true);
    }
    
    private <T> void appendIfNotNull(String name, Object o, boolean newline) throws IOException {
        if (o != null) {
            if (newline) { forceNewline(); }                
            g.write(name, toJson(o));
        }
    }

    private static JsonValue toJson(Object o) {
        if (o == null) {
            return JsonValue.NULL;
        } else if (o instanceof JsonValue) {
            return (JsonValue) o;
        } else if (o instanceof List) {
            List<?> l = (List<?>) o;
            JsonArrayBuilder b = Json.createArrayBuilder();
            for (int i=0; i<l.size(); i++) {
                b.add(toJson(l.get(i)));
            }
            return b.build();
        } else if (o instanceof int[]) {
            int[] l = (int[]) o;
            JsonArrayBuilder b = Json.createArrayBuilder();
            for (int i=0; i<l.length; i++) {
                b.add(l[i]);
            }
            return b.build();
        } else if (o instanceof String) {
            return safeStringToJson((String)o);
        } else if (o instanceof Enum) {
            return toJson(((Enum<?>)o).name());
        } else {
            throw new RuntimeException("Unsupported type: " + o.getClass());
        }
    }
    
    /* ------------ Object Specific Transformations -------------- */

    public static JsonArray nePairsToJson(List<Pair<NerMention, NerMention>> nePairs) {
        JsonArrayBuilder pairs = Json.createArrayBuilder();
        for (Pair<NerMention, NerMention> nePair : nePairs) {
            pairs.add(Json.createObjectBuilder()
                    .add("m1", nemToJson(nePair.get1()))
                    .add("m2", nemToJson(nePair.get2())));
        }
        return pairs.build();
    }
    
    public static JsonArray nesToJson(NerMentions nes) {
        JsonArrayBuilder jnes = Json.createArrayBuilder();
        for (NerMention ne : nes) {
            jnes.add(nemToJson(ne));
        }
        return jnes.build();
    }
    
    private static JsonValue nemToJson(NerMention nem) {
        return Json.createObjectBuilder()
                .add("start", nem.getSpan().start())
                .add("end", nem.getSpan().end())
                .add("head", nem.getHead())
                .add("type", safeStringToJson(nem.getEntityType()))
                .add("subtype", safeStringToJson(nem.getEntitySubType()))
                .add("phraseType", safeStringToJson(nem.getPhraseType()))
                .add("id", safeStringToJson(nem.getId()))
                .build();
    }

    public static List<Pair<NerMention,NerMention>> nePairsFromJson(String json) {
        JsonReader jsonReader = Json.createReader(new StringReader(json));
        JsonArray pairs = jsonReader.readArray();
        jsonReader.close();
        
        List<Pair<NerMention,NerMention>> nePairs = new ArrayList<>();
        for (int i=0; i<pairs.size(); i++) {
            JsonObject pair = pairs.getJsonObject(i);
            nePairs.add(new Pair<NerMention,NerMention>(
                    nemFromJson(pair.getJsonObject("m1")),
                    nemFromJson(pair.getJsonObject("m2"))));
        }
        return nePairs;
    }

    /**
     * Converts from JSON to NerMentions.
     * 
     * @param json The input JSON string.
     * @param n The number of words in the sentence.
     * @return The named entity mentions.
     */
    public static NerMentions nesFromJson(String json, int n) {
        JsonReader jsonReader = Json.createReader(new StringReader(json));
        JsonArray pairs = jsonReader.readArray();
        jsonReader.close();
        
        List<NerMention> spans = new ArrayList<>();
        for (int i=0; i<pairs.size(); i++) {
            JsonObject ne = pairs.getJsonObject(i);
            spans.add(nemFromJson(ne));
        }
        NerMentions nes = new NerMentions(n, spans);
        return nes;
    }

    private static NerMention nemFromJson(JsonObject m) {
        return new NerMention(
                new Span(m.getInt("start"), m.getInt("end")), 
                safeJsonToString(m, "type"),
                safeJsonToString(m, "subtype"),
                safeJsonToString(m, "phraseType"),
                m.getInt("head"), 
                safeJsonToString(m, "id"));
    }
    
    private static String safeJsonToString(JsonObject m, String key) {
        return m.get(key) == JsonValue.NULL ? null : m.getString(key);
    }

    private static JsonValue safeStringToJson(String s) {
        return (s == null) ? JsonValue.NULL : Json.createObjectBuilder().add("temp", s).build().getJsonString("temp");
    }
    
}
