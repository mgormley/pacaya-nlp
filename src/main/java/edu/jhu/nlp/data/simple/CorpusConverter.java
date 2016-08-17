package edu.jhu.nlp.data.simple;

import java.io.IOException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.AnnoPipeline;
import edu.jhu.nlp.ner.NerRunner;
import edu.jhu.nlp.relations.RelationMunger;
import edu.jhu.nlp.relations.RelationMunger.RelationDataPostproc;
import edu.jhu.nlp.relations.RelationMunger.RelationDataPreproc;
import edu.jhu.nlp.relations.RelationMunger.RelationMungerPrm;
import edu.jhu.pacaya.util.cli.ArgParser;
import edu.jhu.pacaya.util.cli.Opt;

/** TODO: make this more transparent:
* Conll data has slots for both gold and automatically predicted POS, FEATS,
* DEPS, and DEPRELS, by --useGoldSyntax True will use the gold and
* --useGoldSyntax False will use the automatic labels. Currently, the
* conll_writer writes the single field in anno_sentence out as both fields.
*
*/

public class CorpusConverter {

    private static final Logger log = LoggerFactory.getLogger(NerRunner.class);

    @Opt(description="Perform relation munging.")
    public static boolean mungeRelations = true;
    
    private static void run(ArgParser parser) throws IOException {
        CorpusHandler handler = new CorpusHandler();
        AnnoSentenceCollection sents = handler.getTrainGold();
        if (mungeRelations) {
            AnnoPipeline prep = new AnnoPipeline();
            RelationMunger relMunger = new RelationMunger(parser.getInstanceFromParsedArgs(RelationMungerPrm.class));
            // Pre-processing.
            RelationDataPreproc dataPreproc = relMunger.getDataPreproc();
            prep.add(dataPreproc);
            if (!relMunger.getPrm().makeRelSingletons) {
                // Post-processing.
                RelationDataPostproc dataPostproc = relMunger.getDataPostproc();
                prep.add(dataPostproc);
            }
            prep.annotate(sents);
            AnnoSentenceReader.logSentStats(sents, log, "train");
        }
        handler.writeTrainGold();
    }
    
    public static void main(String[] args) throws IOException {
        ArgParser parser = new ArgParser(CorpusConverter.class);
        parser.registerClass(CorpusConverter.class);
        parser.registerClass(CorpusHandler.class);
        parser.registerClass(RelationMungerPrm.class);
        parser.parseArgs(args);

        CorpusConverter.run(parser);
    }
    
}
