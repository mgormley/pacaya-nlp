package edu.jhu.nlp.data.simple;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.data.NerMention;
import edu.jhu.nlp.data.semeval.SemEval2010Reader;
import edu.jhu.nlp.tag.StrictPosTagAnnotator.StrictPosTag;
import edu.jhu.pacaya.parse.cky.data.NaryTree;
import edu.jhu.prim.tuple.Pair;

public class SimpleTextReader implements CloseableIterable<AnnoSentence>, Iterator<AnnoSentence> {
    
    private static final Logger log = LoggerFactory.getLogger(SemEval2010Reader.class);
    private AnnoSentence sentence;
    private BufferedReader reader;

    public SimpleTextReader(File file) throws IOException {
        this(new FileInputStream(file));
    }

    public SimpleTextReader(InputStream inputStream) throws UnsupportedEncodingException {
        this(new BufferedReader(new InputStreamReader(inputStream, "UTF-8")));
    }

    public SimpleTextReader(BufferedReader reader) {
        this.reader = reader;
        next();
    }

    private static final Pattern LINE_RE = Pattern.compile("([^:]+): (.*)");
    private static final Pattern SPACE_RE = Pattern.compile(" ");
    private static final Pattern KEY_VAL_RE = Pattern.compile("(\\S+)=(\\S+)");
    
    public static AnnoSentence readSentence(BufferedReader reader) throws IOException {
        // Collect the lines into a list.
        ArrayList<String> lines = new ArrayList<>();
        {
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.equals("")) {
                    break;
                }
                lines.add(line);
            }       
            if (lines.isEmpty()) {
                return null;
            }
        }
        
        // Process each line in this sentence.
        AnnoSentence sent = new AnnoSentence();
        for (String line : lines) {
            Matcher matcher = LINE_RE.matcher(line);
            if (!matcher.matches()) {
                throw new RuntimeException("Unsupport line format: " + line);
            }
            String lineType = matcher.group(1);
            String lineValue = matcher.group(2);
            
            if (lineType.equals("words")) {
                sent.setWords(Arrays.asList(SPACE_RE.split(lineValue)));
            } else if (lineType.equals("prefixes")) {
                sent.setPrefixes(Arrays.asList(SPACE_RE.split(lineValue)));
            } else if (lineType.equals("lemmas")) {
                sent.setLemmas(Arrays.asList(SPACE_RE.split(lineValue)));
            } else if (lineType.equals("posTags")) {
                sent.setPosTags(Arrays.asList(SPACE_RE.split(lineValue)));
            } else if (lineType.equals("cposTags")) {
                sent.setCposTags(Arrays.asList(SPACE_RE.split(lineValue)));
            } else if (lineType.equals("strictPosTags")) {
                List<String> tags = Arrays.asList(SPACE_RE.split(lineValue));
                sent.setStrictPosTags(tags.stream().map(x -> StrictPosTag.valueOf(x)).collect(Collectors.toList()));
            } else if (lineType.equals("clusters")) {
                sent.setClusters(Arrays.asList(SPACE_RE.split(lineValue)));
            } else if (lineType.equals("chunks")) {
                sent.setChunks(Arrays.asList(SPACE_RE.split(lineValue)));
            } else if (lineType.equals("neTags")) {
                sent.setNeTags(Arrays.asList(SPACE_RE.split(lineValue)));
            } else if (lineType.equals("parents")) {
                List<String> rents = Arrays.asList(SPACE_RE.split(lineValue));
                int[] parents = new int[rents.size()];
                for (int i=0; i<rents.size(); i++) {
                    parents[i] = Integer.valueOf(rents.get(i));
                }
                sent.setParents(parents);
            } else if (lineType.equals("deprels")) {
                sent.setDeprels(Arrays.asList(SPACE_RE.split(lineValue)));
            } else if (lineType.equals("naryTree")) {
                sent.setNaryTree(NaryTree.fromTreeInPtbFormat(lineValue));
            } else if (lineType.equals("nePairs")) {
                sent.setNePairs(SimpleTextWriter.nePairsFromJson(lineValue));
            } else if (lineType.equals("relLabels")) {
                sent.setRelLabels(Arrays.asList(SPACE_RE.split(lineValue)));
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
                throw new RuntimeException("Unsupported line type (" + lineType + ") in line: " + line);
            }
        }
        
        return sent;
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
