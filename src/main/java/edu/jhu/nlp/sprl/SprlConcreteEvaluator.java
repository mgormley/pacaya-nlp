package edu.jhu.nlp.sprl;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.commons.cli.ParseException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.AbstractParallelAnnotator;
import edu.jhu.nlp.data.Properties;
import edu.jhu.nlp.data.Properties.Property;
import edu.jhu.nlp.data.Span;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.data.simple.AnnoSentenceCollection;
import edu.jhu.nlp.data.simple.AnnoSentenceReader;
import edu.jhu.nlp.data.simple.AnnoSentenceReader.AnnoSentenceReaderPrm;
import edu.jhu.nlp.data.simple.AnnoSentenceReader.DatasetType;
import edu.jhu.nlp.eval.SprlEvaluator;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.RoleStructure;
import edu.jhu.pacaya.util.cli.ArgParser;
import edu.jhu.pacaya.util.cli.Opt;
import edu.jhu.prim.tuple.Pair;
import edu.jhu.prim.tuple.Quadruple;
import edu.jhu.prim.tuple.Triple;

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

    @Opt(hasArg = true, description = "output file for examples")
    public static File examplesOut= null;

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

    private static List<SprlClassLabel> propList(AnnoSentence a, Pair<Integer, Integer> pair) {
        return a.getSprl().get(pair).toLabels();
    }

    public static void findSprlExamples(AnnoSentenceCollection gold, AnnoSentenceCollection pred, File outFile) {
        // I want to find two pred-arg pairs such that they are in two different
        // sentences, they have the same predicate and the same gold SRL label,
        // but different predicted sprl and different gold sprl
        // for each satisfying pair, I can compute the F1 of the sprl for just
        // that pair and then return the list of these sorted by F1
        // build a list of Pair<PairF1, Triple<AnnoSentence, PredIndex,
        // ArgIndex>> o those pairs that satisfy the constraints
        // first find all pairs that match a particular predicate, argLabel
        // Map[(predWord, argLabel) -> Map[(sentenceId) ->
        // list[(pred,arg,goldSprl,predSprl)]]]
        int nSentences = pred.size();
        Map<Pair<String, String>, Map<Integer, List<Quadruple<Integer, Pair<Integer, Integer>, List<SprlClassLabel>, List<SprlClassLabel>>>>> sprlListsByPredAndSrl = new HashMap<>();
        for (int i = 0; i < nSentences; i++) {
            AnnoSentence g = gold.get(i);
            AnnoSentence p = pred.get(i);
            for (Pair<Integer, Integer> pair : g.getKnownSprlPairs()) {
                int predIx = pair.get1();
                int argIx = pair.get2();
                String predWord = g.getWord(predIx);
                String argLabel = g.getSrlGraph().getEdge(predIx, argIx).getLabel();
                Pair<String, String> predAndSrl = new Pair<>(predWord, argLabel);
                Map<Integer, List<Quadruple<Integer, Pair<Integer, Integer>, List<SprlClassLabel>, List<SprlClassLabel>>>> sentToPairs = sprlListsByPredAndSrl
                        .get(predAndSrl);
                if (sentToPairs == null) {
                    sentToPairs = new HashMap<>();
                    sprlListsByPredAndSrl.put(predAndSrl, sentToPairs);
                }
                List<Quadruple<Integer, Pair<Integer, Integer>, List<SprlClassLabel>, List<SprlClassLabel>>> pairs = sentToPairs
                        .get(i);
                if (pairs == null) {
                    pairs = new LinkedList<>();
                    sentToPairs.put(i, pairs);
                }
                pairs.add(new Quadruple<>(i, pair, propList(g, pair), propList(p, pair)));
            }
        }

        // score, <sentence, pred, arg>
        List<Triple<Double, Quadruple<Integer, Pair<Integer, Integer>, List<SprlClassLabel>, List<SprlClassLabel>>, Quadruple<Integer, Pair<Integer, Integer>, List<SprlClassLabel>, List<SprlClassLabel>>>> scoredExamples = new ArrayList<>();
        // now collect the scores of each by type
        for (Pair<String, String> predAndSrl : sprlListsByPredAndSrl.keySet()) {
            Map<Integer, List<Quadruple<Integer, Pair<Integer, Integer>, List<SprlClassLabel>, List<SprlClassLabel>>>> sentToPairs = sprlListsByPredAndSrl
                    .get(predAndSrl);
            for (Integer sent1Ix : sentToPairs.keySet()) {
                for (Quadruple<Integer, Pair<Integer, Integer>, List<SprlClassLabel>, List<SprlClassLabel>> pair1 : sentToPairs
                        .get(sent1Ix)) {
                    for (Integer sent2Ix : sentToPairs.keySet()) {
                        if (sent2Ix <= sent1Ix) {
                            // skip matches and avoid getting both orders
                            continue;
                        }
                        for (Quadruple<Integer, Pair<Integer, Integer>, List<SprlClassLabel>, List<SprlClassLabel>> pair2 : sentToPairs
                                .get(sent2Ix)) {
                            // only keep if the gold and predicted properties
                            // don't match
                            if (pair1.get3().equals(pair2.get3()) || pair1.get4().equals(pair2.get4())) {
                                continue;
                            }
                            // compute the score as the harmonic mean o the two
                            // F1's
                            double pair1F1 = getF1(pair1.get3(), pair1.get4());
                            double pair2F1 = getF1(pair2.get3(), pair1.get4());
                            double score = ConfusionMatrix.harmonicMean(pair1F1, pair2F1);
                            scoredExamples.add(new Triple<>(score, pair1, pair2));
                        }
                    }

                }
            }
        }

        scoredExamples.sort(Comparator.comparingDouble(t -> -t.get1()));

        // write the confusion map to a file
        try {
            log.info(String.format("Writing examples to: %s", outFile));
            Writer fw = new PrintWriter(outFile);
            
            int nExamples = 10;
            int i = 0;
            for (Triple<Double, Quadruple<Integer, Pair<Integer, Integer>, List<SprlClassLabel>, List<SprlClassLabel>>, Quadruple<Integer, Pair<Integer, Integer>, List<SprlClassLabel>, List<SprlClassLabel>>> example : scoredExamples) {
                printExample(gold, example.get2(), fw);
                printExample(gold, example.get3(), fw);
                fw.write("\n");
                i++;
                if (i >= nExamples) {
                    break;
                }
            }
            fw.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void printExample(AnnoSentenceCollection gold, 
            Quadruple<Integer, Pair<Integer, Integer>, List<SprlClassLabel>, List<SprlClassLabel>> example, Writer fw) throws IOException {
        int i = example.get1();
        Pair<Integer, Integer> pair = example.get2();
        AnnoSentence g = gold.get(i);
        fw.write(g.getWordsStr(new Span(0, g.size())));
        fw.write("\n");
        int predIx = pair.get1();
        int argIx = pair.get2();
        fw.write(String.format("Predicate at %s: %s\n", predIx, g.getWord(predIx)));
        fw.write(String.format("Argument at %s (%s): %s\n", argIx, g.getWord(argIx),
                g.getSrlGraph().getEdge(predIx, argIx).getLabel()));
        fw.write(String.format("%30s\t%15s\t%15s\n", "Property", "Gold", "Predicted"));
        for (Property q : Property.values()) {
            fw.write(String.format("%30s\t%15s\t%15s\n", q, example.get3().get(q.ordinal()), example.get4().get(q.ordinal())));
        }
        fw.write("\n");
    }

    private static double getF1(List<SprlClassLabel> gold, List<SprlClassLabel> pred) {
        Set<SprlClassLabel> nils = SprlClassLabel.getNils();
        ConfusionMap<SprlClassLabel, Property> cms = new ConfusionMap<SprlClassLabel, Properties.Property>(nils);
        for (int i = 0; i < gold.size(); i++) {
            cms.recordPrediction(gold.get(i), pred.get(i), Property.values()[i]);
        }
        return cms.getTotal().f1();
    }

    public static void evalSprl(AnnoSentenceCollection gold, AnnoSentenceCollection pred) {
        Set<SprlClassLabel> nils = SprlClassLabel.getNils();
        ConfusionMap<SprlClassLabel, Property> cms = new ConfusionMap<SprlClassLabel, Properties.Property>(nils);
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
                    cms.recordPrediction(SprlClassLabel.valueOf(gLabels.get(j)), SprlClassLabel.valueOf(pLabels.get(j)),
                            q);
                }
            }
        }

        // set the order
        List<SprlClassLabel> labelOrder = new LinkedList<>(Arrays.asList(SprlClassLabel.LIKELY, SprlClassLabel.UNKNOWN,
                SprlClassLabel.UNLIKELY, SprlClassLabel.NA, SprlClassLabel.NOT_AN_ARG));

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
            if (examplesOut != null) {
                findSprlExamples(goldSents, predSents, examplesOut);
            }
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
