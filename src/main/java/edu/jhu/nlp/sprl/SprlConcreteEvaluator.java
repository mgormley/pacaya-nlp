package edu.jhu.nlp.sprl;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.Writer;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

import org.apache.commons.cli.ParseException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.AbstractParallelAnnotator;
import edu.jhu.nlp.data.Properties;
import edu.jhu.nlp.data.Properties.Property;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.data.simple.AnnoSentenceCollection;
import edu.jhu.nlp.data.simple.AnnoSentenceReader;
import edu.jhu.nlp.data.simple.AnnoSentenceReader.AnnoSentenceReaderPrm;
import edu.jhu.nlp.data.simple.AnnoSentenceReader.DatasetType;
import edu.jhu.nlp.eval.SprlEvaluator;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.RoleStructure;
import edu.jhu.pacaya.util.cli.ArgParser;
import edu.jhu.pacaya.util.cli.Opt;

public class SprlConcreteEvaluator {

    private static final Logger log = LoggerFactory.getLogger(SprlConcreteEvaluator.class);

    @Opt(hasArg = true, description = "concrete file containing predicted sprl judgments")
    public static File pred = null;

    @Opt(hasArg = true, description = "concrete tool to use for predicted sprl judgements")
    public static String predTool = null;

    @Opt(hasArg = true, description = "concrete file containing gold sprl judgments")
    public static File gold = null;

    @Opt(hasArg = true, description = "concrete tool to use for gold sprl judgements")
    public static String goldTool = null;

    @Opt(hasArg = true, description = "output file for confusion matrices")
    public static File outFile = null;

    @Opt(hasArg = true, description = "The structure of the Role variables.")
    public static RoleStructure roleStructure = RoleStructure.PREDS_GIVEN;

    @Opt(hasArg = true, description = "Whether to allow a predicate to assign a role to itself. (This should be turned on for English)")
    public static boolean allowPredArgSelfLoops = true;

    private static AnnoSentenceCollection loadSents(String name, File comm, String tool) throws IOException {
        AnnoSentenceReaderPrm prm = new AnnoSentenceReaderPrm();
        prm.name = name;
        prm.rePrm.depParseTool = null;
        prm.rePrm.srlTool = null;
        prm.rePrm.sprlTool = tool;
        // prm.maxNumSentences = trainMaxNumSentences;
        // prm.maxSentenceLength = trainMaxSentenceLength;
        // prm.minSentenceLength = trainMinSentenceLength;
        // prm.useCoNLLXPhead = trainUseCoNLLXPhead;
        // prm.normalizeRoleNames = normalizeRoleNames;
        // prm.useGoldSyntax = useGoldSyntax;
        AnnoSentenceReader reader = new AnnoSentenceReader(prm);
        reader.loadSents(comm, DatasetType.CONCRETE);
        return reader.getData();
    }

    public static enum EvalType {
        AllPairs, KnownPreds, KnownPairs,
    }

    public static void evalSprl(AnnoSentenceCollection gold, AnnoSentenceCollection pred) {
        ConfusionMap<String, Property> cms = new ConfusionMap<String, Properties.Property>(
                SprlClassLabel.NOT_AN_ARG.name());
        Set<SprlClassLabel> nils = SprlClassLabel.getNils();
        int nSentences = pred.size();

        for (int i = 0; i < nSentences; i++) {
            AnnoSentence g = gold.get(i);
            AnnoSentence p = pred.get(i);
            for (Property q : Property.values()) {
                SprlEvaluator eval = new SprlEvaluator(roleStructure, allowPredArgSelfLoops, nils, q);
                List<String> gLabels = eval.getLabels(g, g);
                List<String> pLabels = eval.getLabels(p, g);
                assert pLabels.size() == gLabels.size();
                for (int j = 0; j < pLabels.size(); j++) {
                    cms.recordPrediction(gLabels.get(j), pLabels.get(j), q);
                }
            }
        }

        // set the order
        List<String> labelOrder = new LinkedList<>(Arrays.asList(SprlClassLabel.LIKELY.name(), SprlClassLabel.UNKNOWN.name(),
                SprlClassLabel.UNLIKELY.name(), SprlClassLabel.NA.name(), SprlClassLabel.NOT_AN_ARG.name()));

        // but only include things we saw
        for (SprlClassLabel k : SprlClassLabel.values()) {
            if (!cms.total.keySet().contains(k.name())) {
                labelOrder.remove(k.name());
            }
        }

        // write the confusion map to a file
        try {
            Writer fw = new PrintWriter(outFile);
            cms.print(labelOrder, fw);
            log.info(String.format("Writing to: %s", outFile.getAbsolutePath()));
            fw.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    public static void main(String[] args) {
        int exitCode = 0;
        ArgParser parser = null;
        try {
            parser = new ArgParser(SprlConcreteEvaluator.class);
            parser.registerClass(SprlConcreteEvaluator.class);
            parser.registerClass(SprlClassLabel.class);
            parser.parseArgs(args);
            AnnoSentenceCollection predSents = loadSents("pred", pred, predTool);
            AnnoSentenceCollection goldSents = loadSents("gold", gold, goldTool);
            evalSprl(goldSents, predSents);
        } catch (ParseException e1) {
            log.error(e1.getMessage());
            if (parser != null) {
                parser.printUsage();
            }
            exitCode = 1;
        } catch (Throwable t) {
            AbstractParallelAnnotator.logThrowable(log, t);
            exitCode = 1;
        }
        System.exit(exitCode);
    }

}
