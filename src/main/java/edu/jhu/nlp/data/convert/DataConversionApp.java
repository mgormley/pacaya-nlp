package edu.jhu.nlp.data.convert;
import java.io.IOException;

import org.apache.commons.cli.ParseException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.data.simple.CorpusHandler;
import edu.jhu.pacaya.util.cli.ArgParser;

/**
 * Uses the CorpusHandler to just read training data and write it to a different
 * format.
 * 
 * Example:
 *
 * java edu.jhu.nlp.data.convert.DataConversionApp --train /home/adam/data/LDC/LDC2012T04/2009_conll_p2/data/CoNLL2009-ST-English/CoNLL2009-ST-English-trial.txt --trainType CONLL_2009 --trainGoldOut ./train.out.conll --trainTypeOut CONLL_2009 --language en --trainProjectivize False --trainUseCoNLLXPhead True --useGoldSyntax False
 * 
 * TODO: make this more transparent:
 * 
 * Conll data has slots for both gold and automatically predicted POS, FEATS,
 * DEPS, and DEPRELS, by --useGoldSyntax True will use the gold and
 * --useGoldSyntax False will use the automatic labels. Currently, the
 * conll_writer writes the single field in anno_sentence out as both fields.
 * 
 */
public class DataConversionApp {

    private static final Logger log = LoggerFactory.getLogger(DataConversionApp.class);

    public static void main(String[] args) throws ParseException, IOException {
        ArgParser parser = new ArgParser(DataConversionApp.class);
        parser.registerClass(CorpusHandler.class);
        parser.parseArgs(args);
        CorpusHandler corpus = new CorpusHandler();
        if (!corpus.hasTrain()) {
            log.error("Requires --train option");
            System.exit(1);
        }
        if (!CorpusHandler.useGoldSyntax) {
            log.warn("not using gold syntax");
        } else {
            log.info("using gold syntax from input");
        }
        corpus.getTrainGold();
        corpus.writeTrainGold();
    }

}
