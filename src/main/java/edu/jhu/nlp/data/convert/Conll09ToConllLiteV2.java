package edu.jhu.nlp.data.convert;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Map;

import org.apache.commons.cli.ParseException;
import org.apache.commons.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.data.conll.CoNLL09Reader;
import edu.jhu.nlp.data.conll.CoNLL09Sentence;
import edu.jhu.nlp.data.conll.CoNLL09Token;
import edu.jhu.nlp.data.conll.SrlGraph;
import edu.jhu.nlp.data.conll.SrlGraph.SrlArg;
import edu.jhu.nlp.data.conll.SrlGraph.SrlEdge;
import edu.jhu.pacaya.util.cli.ArgParser;
import edu.jhu.pacaya.util.cli.Opt;
import edu.jhu.prim.util.random.Prng;

/**
 * Converts CoNLL-2009 format to CoNLL Lite (with a few extra columns).
 *  
 * @author mgormley
 *
 */
public class Conll09ToConllLiteV2 {

    private static final Logger log = LoggerFactory.getLogger(Conll09ToConllLiteV2.class);

    @Opt(hasArg = true, required = true, description = "CoNLL 09 input file")
    public static File input;
    @Opt(hasArg = true, required = true, description = "CoNLL Lite output file")
    public static File output;
    @Opt(hasArg = true, description = "Pseudo random number generator seed")
    public static long seed = Prng.DEFAULT_SEED;

    public void run() throws IOException {
        final String sep = "\t";

        BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(output), "UTF-8"));
        CoNLL09Reader reader = new CoNLL09Reader(input);
        for (CoNLL09Sentence sent : reader) {
            SrlGraph srl = sent.getSrlGraph();
            
            Map<Integer, SrlArg> posToArgMap = srl.getPositionArgMap();
            
            int i=0; 
            for (CoNLL09Token tok : sent) {
                ArrayList<String> cols = new ArrayList<String>();
                cols.add(Integer.toString(tok.getId()));
                cols.add(tok.getForm());
                cols.add(tok.getLemma());
                cols.add(tok.getPos());
                cols.add(Integer.toString(tok.getHead()));
                
                SrlArg arg = posToArgMap.get(i);
                if (arg != null) {
                    for (SrlEdge edge : arg.getEdges()) {
                        String eStr = "";
                        eStr += edge.getPred().getId();
                        eStr += ", ";
                        eStr += edge.getLabel();
                        cols.add(eStr);
                    }
                }
                
                writer.write(StringUtils.join(cols, sep) + "\n");
                i++;
            }
            writer.write("\n");
        }
        
        reader.close();
        writer.close();
    }
    
    public static void main(String[] args) throws IOException {
        ArgParser parser = new ArgParser(Conll09ToConllLiteV2.class);
        parser.registerClass(Conll09ToConllLiteV2.class);
        parser.parseArgs(args);
        
        Prng.seed(seed);
        
        Conll09ToConllLiteV2 pipeline = new Conll09ToConllLiteV2();
        pipeline.run();
    }

}
