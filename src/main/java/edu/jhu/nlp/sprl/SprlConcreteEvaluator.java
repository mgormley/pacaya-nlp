package edu.jhu.nlp.sprl;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringReader;
import java.io.StringWriter;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.IntStream;

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
    public static File examplesOut = null;

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
        List<Property> propertyOrder = Arrays.asList(Property.values());
        // write the confusion map to a file
        try {
            log.info(String.format("Writing examples to: %s", outFile));
            Writer fw = new PrintWriter(outFile);

            int nExamples = 10;
            int i = 0;
            for (Triple<Double, Quadruple<Integer, Pair<Integer, Integer>, List<SprlClassLabel>, List<SprlClassLabel>>, Quadruple<Integer, Pair<Integer, Integer>, List<SprlClassLabel>, List<SprlClassLabel>>> example : scoredExamples) {
                Quadruple<Integer, Pair<Integer, Integer>, List<SprlClassLabel>, List<SprlClassLabel>> ex1 = example
                        .get2();
                AnnoSentence s1 = gold.get(ex1.get1());
                String ex1Str = formatExample(s1, ex1.get2(), ex1.get3(), ex1.get4(), propertyOrder);

                Quadruple<Integer, Pair<Integer, Integer>, List<SprlClassLabel>, List<SprlClassLabel>> ex2 = example
                        .get3();
                AnnoSentence s2 = gold.get(ex2.get1());
                String ex2Str = formatExample(s2, ex2.get2(), ex2.get3(), ex2.get4(), propertyOrder);

                fw.write(String.format("%s\n", getSentence(s1)));
                fw.write(String.format("%s\n", getSentence(s2)));
                fw.write(String.format("%s\n", hStack(ex1Str, ex2Str)));
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

    /**
     * does not include the line terminating characters
     * 
     * @param s
     * @return
     */
    private static List<String> getLines(String s) {
        // adapted from
        // http://stackoverflow.com/questions/13464954/how-do-i-split-a-string-by-line-break
        List<String> lines = new ArrayList<String>();
        try {
            BufferedReader rdr = new BufferedReader(new StringReader(s));
            for (String line = rdr.readLine(); line != null; line = rdr.readLine()) {
                lines.add(line);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return lines;
    }

    private static int maxWidth(List<String> lines) {
        int w = 0;
        for (String line : lines) {
            w = Math.max(w, line.length());
        }
        return w;
    }

    private static String getLine(List<String> lines, int i) {
        if (i < lines.size()) {
            return lines.get(i);
        } else {
            return "";
        }
    }

    /**
     * Horizontally stack the lines of the two strings
     */
    public static String hStack(String... a) {
        List<List<String>> lines = new ArrayList<List<String>>(a.length);
        List<Integer> maxWidths = new ArrayList<Integer>(a.length);
        int totalWidth = 0;
        int maxLines = 0;
        for (String s : a) {
            List<String> sLines = getLines(s);
            lines.add(sLines);

            // keep track of how wide each bloack is
            maxWidths.add(maxWidth(sLines));

            // keep track of the total max width
            totalWidth += 0;

            // keep track of how many total lines
            maxLines = Math.max(maxLines, sLines.size());
        }
        
        // now paste the blocks together
        StringWriter sw = new StringWriter(maxLines * (totalWidth + 2));
        for (int i = 0; i < maxLines; i++) {
            // for each line number, print out the corresponding line from each
            // string
            for (int j = 0; j < maxWidths.size(); j++) {
                int maxW = maxWidths.get(j);
                if (maxW > 0) {
                    // for the jth string, print the ith line with the max width
                    String formatString = "%-" + maxW + "s";
                    String line = getLine(lines.get(j), i);
                    sw.write(String.format(formatString, line));
                }
            }
            sw.write("\n");
        }
        return sw.toString();
    }

    private static String getSentence(AnnoSentence s) {
        return s.getWordsStr(new Span(0, s.size()));
    }

    private static String formatExample(AnnoSentence gold, Pair<Integer, Integer> pair, List<SprlClassLabel> goldLabels,
            List<SprlClassLabel> predicatedLabels, List<Property> propertyOrder) {
        StringWriter sw = new StringWriter();
        int predIx = pair.get1();
        int argIx = pair.get2();
        sw.write(String.format("Predicate at %s: %s\n", predIx, gold.getWord(predIx)));
        sw.write(String.format("Argument at %s (%s): %s\n", argIx, gold.getWord(argIx),
                gold.getSrlGraph().getEdge(predIx, argIx).getLabel()));
        sw.write(String.format("%30s\t%15s\t%15s\n", "Property", "Gold", "Predicted"));
        for (int q = 0; q < propertyOrder.size(); q++) {
            sw.write(String.format("%30s\t%15s\t%15s\n", propertyOrder.get(q), goldLabels.get(q),
                    predicatedLabels.get(q)));
        }
        sw.write("\n");
        return sw.toString();
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
            SprlEvaluator eval = new SprlEvaluator(roleStructure, allowPredArgSelfLoops, nils);
            List<Triple<Integer, Integer, Property>> examples = eval.getExamples(p, g);
            List<String> gLabels = eval.getLabels(g, g);
            List<String> pLabels = eval.getLabels(p, g);
            assert gLabels.size() == pLabels.size() && gLabels.size() == examples.size();
            for (int x = 0; x < examples.size(); x++) {
                Triple<Integer, Integer, Property> example = examples.get(x);
                SprlClassLabel gL = SprlClassLabel.valueOf(gLabels.get(x));
                SprlClassLabel pL = SprlClassLabel.valueOf(pLabels.get(x));
                Property q = example.get3();
                int predIx = example.get1();
                int argIx = example.get2();
                String exStr = cms.hasExample(gL, pL, q) ? null
                        : String.format("%s\n%s\n", getSentence(g), formatExample(g, new Pair<>(predIx, argIx), Collections.singletonList(gL),
                                Collections.singletonList(pL), Collections.singletonList(q)));
                cms.recordPrediction(gL, pL, q, exStr);
            }
        }

        // set the order
        List<SprlClassLabel> labelOrder = new LinkedList<>(Arrays.asList(SprlClassLabel.LIKELY, SprlClassLabel.UNKNOWN,
                SprlClassLabel.UNLIKELY, SprlClassLabel.NA, SprlClassLabel.NOT_AN_ARG));

        // but only include things we saw
        for (SprlClassLabel k : SprlClassLabel.values()) {
            if (!cms.getTotal().keySet().contains(k.name())) {
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
