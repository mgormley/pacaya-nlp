package edu.jhu.nlp.data.simple;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.data.concrete.ConcreteReader;
import edu.jhu.nlp.data.concrete.ConcreteReader.ConcreteReaderPrm;
import edu.jhu.nlp.data.concrete.ListCloseableIterable;
import edu.jhu.nlp.data.conll.CoNLL02Reader;
import edu.jhu.nlp.data.conll.CoNLL02Sentence;
import edu.jhu.nlp.data.conll.CoNLL03Reader;
import edu.jhu.nlp.data.conll.CoNLL03Sentence;
import edu.jhu.nlp.data.conll.CoNLL08Reader;
import edu.jhu.nlp.data.conll.CoNLL08Sentence;
import edu.jhu.nlp.data.conll.CoNLL09Reader;
import edu.jhu.nlp.data.conll.CoNLL09Sentence;
import edu.jhu.nlp.data.conll.CoNLLXReader;
import edu.jhu.nlp.data.conll.CoNLLXSentence;
import edu.jhu.nlp.data.semeval.SemEval2010Reader;
import edu.jhu.nlp.data.semeval.SemEval2010Sentence;

/**
 * Generic reader of AnnoSentence objects from many different corpora. 
 * 
 */
public class AnnoSentenceReader {

    public static class AnnoSentenceReaderPrm {
        public boolean useGoldSyntax = false;
        public int maxNumSentences = Integer.MAX_VALUE; 
        public int maxSentenceLength = Integer.MAX_VALUE; 
        public int minSentenceLength = 0;
        public SentFilter filter = null;
        public String name = "";
        
        // Parameters specific to data set type.
        /** CoNLL-2009 / CoNLL-2008: Whether to normalize role names in SRL data. */
        public boolean normalizeRoleNames = false;
        /** CoNLL-2008: Whether to use split word forms. */
        public boolean useSplitForms = true;
        /** CoNLL-X: whether to use the P(rojective)HEAD column for parents. */
        public boolean useCoNLLXPhead = false;
        /** Concrete options. */
        public ConcreteReaderPrm rePrm = new ConcreteReaderPrm();        
    }

    public enum DatasetType {
        SYNTHETIC, PTB, CONLL_2002, CONLL_2003, CONLL_X, CONLL_2008, CONLL_2009, 
        CONCRETE, SEMEVAL_2010, DEP_EDGE_MASK, JSON
    };

    public interface SASReader extends Iterable<AnnoSentence> {
        public void close();        
    }
    
    private static final Logger log = LoggerFactory.getLogger(AnnoSentenceReader.class);

    private AnnoSentenceReaderPrm prm;
    private AnnoSentenceCollection sents;
    
    public AnnoSentenceReader(AnnoSentenceReaderPrm prm) {
        this.prm = prm;
        this.sents = new AnnoSentenceCollection();
    }
    
    public AnnoSentenceCollection getData() {
        return sents;
    }
    
    public void loadSents(File dataFile, DatasetType type) throws IOException {
        log.info("Reading " + prm.name + " data of type " + type + " from " + dataFile);
        if (type == DatasetType.CONCRETE && (dataFile.isDirectory() || dataFile.getName().endsWith(".zip"))) {
            ConcreteReader cr = new ConcreteReader(prm.rePrm);
            AnnoSentenceCollection csents = cr.sentsFromPath(dataFile);
            CloseableIterable<AnnoSentence> reader = new ListCloseableIterable(csents);
            loadSents(reader);
            sents.setSourceSents(csents.getSourceSents());
            reader.close();
            logSentStats(sents, log, prm.name);
        } else {
            InputStream fis = new FileInputStream(dataFile);
            loadSents(fis, type);
            fis.close();
        }
    }
    
    public void loadSents(InputStream fis, DatasetType type) throws IOException {
        if (prm.normalizeRoleNames) {
            if (type == DatasetType.CONLL_2008 || type == DatasetType.CONLL_2009) {
                log.info("Normalizing role names");
            }
        }
        
        CloseableIterable<AnnoSentence> reader = null;
        Object sourceSents = null;
        if (type == DatasetType.CONCRETE) {
            ConcreteReader cr = new ConcreteReader(prm.rePrm);
            AnnoSentenceCollection csents = cr.sentsFromCommInputStream(fis);
            sourceSents = csents.getSourceSents();
            reader = new ListCloseableIterable(csents);
        } else {
            if (type == DatasetType.CONLL_2009) {
                reader = ConvCloseableIterable.getInstance(new CoNLL09Reader(fis), new CoNLL092Anno());
            } else if (type == DatasetType.CONLL_2008) {
                reader = ConvCloseableIterable.getInstance(new CoNLL08Reader(fis), new CoNLL082Anno());
            } else if (type == DatasetType.CONLL_X) {
                reader = ConvCloseableIterable.getInstance(new CoNLLXReader(fis), new CoNLLX2Anno());
            } else if (type == DatasetType.CONLL_2002) {
                reader = ConvCloseableIterable.getInstance(new CoNLL02Reader(fis), new CoNLL022Anno());
            } else if (type == DatasetType.CONLL_2003) {
                reader = ConvCloseableIterable.getInstance(new CoNLL03Reader(fis), new CoNLL032Anno());
            } else if (type == DatasetType.SEMEVAL_2010) {
                reader = ConvCloseableIterable.getInstance(new SemEval2010Reader(fis), new SemEval20102Anno());
            } else if (type == DatasetType.JSON) {
                reader = new JsonConcatReader(fis);
            //} else if (type == DatasetType.PTB) {
                //reader = new Ptb2Anno(new PtbFileReader(dataFile));
            } else {
                throw new IllegalStateException("Unsupported data type: " + type);
            }
        }
        
        loadSents(reader);
        sents.setSourceSents(sourceSents);
        reader.close();
        logSentStats(sents, log, prm.name);
    }

    public static void logSentStats(AnnoSentenceCollection sents, Logger log, String name) {
        log.info("Num " + name + " sentences: " + sents.size());   
        log.info("Num " + name + " tokens: " + sents.getNumTokens());
        log.info("Longest sentence: " + sents.getMaxLength());
        log.info("Average sentence length: " + sents.getAvgLength());
    }
    
    private void loadSents(Iterable<AnnoSentence> reader) {
        for (AnnoSentence sent : reader) {
            if (sents.size() >= prm.maxNumSentences) {
                break;
            }
            if (sent.size() <= prm.maxSentenceLength && prm.minSentenceLength <= sent.size()) {
                if (prm.filter == null || prm.filter.accept(sent)) {
                    sent.intern();
                    sents.add(sent);
                }
            }
        }
    }
    
    public class CoNLL092Anno implements Converter<CoNLL09Sentence, AnnoSentence> {

        @Override
        public AnnoSentence convert(CoNLL09Sentence x) {
            if (prm.normalizeRoleNames) {
                x.normalizeRoleNames();
            }
            return x.toAnnoSentence(prm.useGoldSyntax);
        }        
        
    }
    
    public class CoNLL082Anno implements Converter<CoNLL08Sentence, AnnoSentence> {

        @Override
        public AnnoSentence convert(CoNLL08Sentence x) {
            if (prm.normalizeRoleNames) {
                x.normalizeRoleNames();
            }
            return x.toAnnoSentence(prm.useGoldSyntax, prm.useSplitForms );
        }
        
    }
    
    public class CoNLLX2Anno implements Converter<CoNLLXSentence, AnnoSentence> {

        @Override
        public AnnoSentence convert(CoNLLXSentence x) {
            return x.toAnnoSentence(prm.useCoNLLXPhead);
        }
        
    }

    public class CoNLL022Anno implements Converter<CoNLL02Sentence, AnnoSentence> {

        @Override
        public AnnoSentence convert(CoNLL02Sentence x) {
            return x.toAnnoSentence();
        }
        
    }

    public class CoNLL032Anno implements Converter<CoNLL03Sentence, AnnoSentence> {

        @Override
        public AnnoSentence convert(CoNLL03Sentence x) {
            return x.toAnnoSentence();
        }
        
    }
    
    public class SemEval20102Anno implements Converter<SemEval2010Sentence, AnnoSentence> {

        @Override
        public AnnoSentence convert(SemEval2010Sentence x) {
            return x.toAnnoSentence();
        }
        
    }
    
}
