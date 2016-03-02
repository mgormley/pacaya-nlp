package edu.jhu.nlp.data.simple;

import java.io.BufferedWriter;
import java.io.Closeable;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.StringReader;
import java.io.StringWriter;
import java.io.Writer;
import java.util.ArrayList;
import java.util.List;

import javax.json.Json;
import javax.json.JsonArray;
import javax.json.JsonArrayBuilder;
import javax.json.JsonObject;
import javax.json.JsonReader;
import javax.json.JsonValue;
import javax.json.JsonWriter;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.data.NerMention;
import edu.jhu.nlp.data.Span;
import edu.jhu.prim.tuple.Pair;

/**
 * Writes a simple text format representing an annotated sentence.
 * 
 * @author mgormley
 */
public class SimpleTextWriter implements Closeable {

    private static final Logger log = LoggerFactory.getLogger(SimpleTextWriter.class);
    private Writer writer;
    private int count;
    
    public SimpleTextWriter(File path) throws IOException {
        this(new FileOutputStream(path));
    }
    
    public SimpleTextWriter(OutputStream os) throws IOException {
        this.writer = new BufferedWriter(new OutputStreamWriter(os, "UTF-8"));
        this.count = 0;
    }
    
    public void write(AnnoSentenceCollection sents) throws IOException {
        for (AnnoSentence sent : sents) { this.write(sent); }
    }
    
    public void write(AnnoSentence sent) throws IOException {
        writer.write(sent2String(sent));
        writer.write("\n");
        count++;
        writer.flush();
    }
    
    public String sent2String(AnnoSentence sent) {
        StringBuilder sb = new StringBuilder();
        appendIfNotNull(sb, "words", sent.getWords());
        appendIfNotNull(sb, "prefixes", sent.getPrefixes());
        appendIfNotNull(sb, "lemmas", sent.getLemmas());
        appendIfNotNull(sb, "posTags", sent.getPosTags());
        appendIfNotNull(sb, "cposTags", sent.getCposTags());
        appendIfNotNull(sb, "strictPosTags", sent.getStrictPosTags());
        appendIfNotNull(sb, "clusters", sent.getClusters());
        appendIfNotNull(sb, "embedIds", sent.getEmbedIds());
        appendIfNotNull(sb, "feats", sent.getFeats());
        appendIfNotNull(sb, "chunks", sent.getChunks());
        appendIfNotNull(sb, "neTags", sent.getNeTags());
        appendIfNotNull(sb, "parents", sent.getParents());
        appendIfNotNull(sb, "deprels", sent.getDeprels());
        appendIfNotNull(sb, "depEdgeMask", sent.getDepEdgeMask());
        appendIfNotNull(sb, "srlGraph", sent.getSrlGraph());
        appendIfNotNull(sb, "knownPreds", sent.getKnownPreds());
        appendIfNotNull(sb, "naryTree", sent.getNaryTree() == null ? null : sent.getNaryTree().getAsOneLineString());
        appendIfNotNull(sb, "namedEntities", sent.getNamedEntities());
        if (sent.getNamedEntities() != null) { 
            appendIfNotNull(sb, "namedEntities-InContext", sent.getNamedEntities().toString(sent.getWords())); 
        }
        appendIfNotNull(sb, "nePairs", nePairsToJson(sent.getNePairs()).toString());
        appendIfNotNull(sb, "relLabels", sent.getRelLabels());
        appendIfNotNull(sb, "relations", sent.getRelations());
        if (sent.getRelations() != null) {
            appendIfNotNull(sb, "relations-InContext", sent.getRelations().toString(sent.getWords())); 
        }
        // Not included:
        // - sent.getSourceSent()
        return sb.toString();
    }

    private static <T> void appendIfNotNull(StringBuilder sb, String name, List<T> strs) {
        if (strs != null) {
            sb.append(name);
            sb.append(": ");
            appendEach(sb, strs, " ");
            sb.append("\n");
        }
    }

    private static void appendIfNotNull(StringBuilder sb, String name, int[] ints) {
        if (ints != null) {
            sb.append(name);
            sb.append(": ");
            appendEach(sb, ints, " ");
            sb.append("\n");
        }
    }

    private static void appendIfNotNull(StringBuilder sb, String name, Object o) {
        if (o != null) {
            sb.append(name);
            sb.append(": ");
            sb.append(o);
            sb.append("\n");
        }
    }

    private static <T> void appendEach(StringBuilder sb, List<T> strs, String sep) {
        for (int i=0; i<strs.size(); i++) {
            if (i != 0) {
                sb.append(sep);
            }
            sb.append(strs.get(i));
        }
    }


    private static void appendEach(StringBuilder sb, int[] ints, String sep) {
        for (int i=0; i<ints.length; i++) {
            if (i != 0) {
                sb.append(sep);
            }
            sb.append(ints[i]);
        }
    }

    public void close() throws IOException {
        writer.close();
    }
    
    public int getCount() {
        return count;
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

    private static NerMention nemFromJson(JsonObject m) {
        return new NerMention(
                new Span(m.getInt("start"), m.getInt("end")), 
                safeGetString(m, "type"),
                safeGetString(m, "subtype"),
                safeGetString(m, "phraseType"),
                m.getInt("head"), 
                safeGetString(m, "id"));
    }
    
    private static String safeGetString(JsonObject m, String key) {
        return m.get(key) == JsonValue.NULL ? null : m.getString(key);
    }

    private static JsonValue safeStringToJson(String s) {
        return (s == null) ? JsonValue.NULL : Json.createObjectBuilder().add("temp", s).build().getJsonString("temp");
    }

    
}
