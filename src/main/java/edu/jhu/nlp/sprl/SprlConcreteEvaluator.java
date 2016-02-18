package edu.jhu.nlp.sprl;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.Writer;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

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
import edu.jhu.nlp.srl.SrlFactorGraphBuilder;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.RoleStructure;
import edu.jhu.pacaya.util.cli.ArgParser;
import edu.jhu.pacaya.util.cli.Opt;
import edu.jhu.prim.tuple.Pair;

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
        ConfusionMap<SprlClassLabel, Property> cms = new ConfusionMap<SprlClassLabel, Properties.Property>(
                SprlClassLabel.NOT_AN_ARG);
        int nSentences = pred.size();
        for (int i = 0; i < nSentences; i++) {
            AnnoSentence goldSent = gold.get(i);
            Map<Pair<Integer, Integer>, Properties> p = pred.get(i).getSprl();
            Map<Pair<Integer, Integer>, Properties> g = goldSent.getSprl();
            for (Pair<Integer, Integer> k : SrlFactorGraphBuilder.getPossibleRolePairs(gold.size(),
                    goldSent.getKnownSprlPreds(), g.keySet(), roleStructure, allowPredArgSelfLoops)) {
                Properties pProps = p.get(k);
                Map<String, Double> pMap = pProps == null ? null : pProps.toMap();
                Properties gProps = g.get(k);
                Map<String, Double> gMap = gProps == null ? null : gProps.toMap();
                for (Property q : Property.values()) {
                    cms.recordPrediction(
                            gProps == null ? SprlClassLabel.NOT_AN_ARG : SprlClassLabel.getLabel(gMap.get(q.name())),
                            pProps == null ? SprlClassLabel.NOT_AN_ARG : SprlClassLabel.getLabel(pMap.get(q.name())),
                            q);
                }
            }
        }

        List<SprlClassLabel> labelOrder = new LinkedList<>(Arrays.asList(SprlClassLabel.LIKELY, SprlClassLabel.UNKNOWN,
                SprlClassLabel.UNLIKELY, SprlClassLabel.NA, SprlClassLabel.NOT_AN_ARG));

        for (SprlClassLabel k : SprlClassLabel.values()) {
            if (!cms.total.keySet().contains(k)) {
                labelOrder.remove(k);
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
