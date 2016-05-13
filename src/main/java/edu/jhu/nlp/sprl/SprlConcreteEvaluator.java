package edu.jhu.nlp.sprl;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.StringReader;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.List;

public class SprlConcreteEvaluator {
//
//    private static final Logger log = LoggerFactory.getLogger(SprlConcreteEvaluator.class);
//    public static final String srlNil = "_";
//
//    @Opt(hasArg = true, description = "the number of examples per confusion matrix cell")
//    public static int numExamples = 2;
//
//    @Opt(hasArg = true, description = "eval sprl")
//    public static boolean includeSprl = true;
//
//    @Opt(hasArg = true, description = "eval srl")
//    public static boolean includeSrl = false;
//
//    @Opt(hasArg = true, description = "concrete file containing predicted sprl judgments")
//    public static File pred = null;
//
//    @Opt(hasArg = true, description = "concrete tool to use for predicted sprl judgements")
//    public static String predTool = null;
//
//    @Opt(hasArg = true, description = "concrete file containing gold sprl judgments")
//    public static File gold = null;
//
//    @Opt(hasArg = true, description = "concrete tool to use for gold sprl judgements")
//    public static String goldTool = null;
//
//    @Opt(hasArg = true, description = "output file for confusion matrices")
//    public static File outFile = null;
//
//    @Opt(hasArg = true, description = "if not null, only pairs with this gold srl label are included")
//    public static String goldSrlLabel = null;
//    
//    @Opt(hasArg = true, description = "output file for examples")
//    public static File examplesOut = null;
//
//    @Opt(hasArg = true, description = "The structure of the Role variables.")
//    public static RoleStructure roleStructure = RoleStructure.PREDS_GIVEN;
//
//    @Opt(hasArg = true, description = "Whether to allow a predicate to assign a role to itself. (This should be turned on for English)")
//    public static boolean allowPredArgSelfLoops = true;
//
//    private static AnnoSentenceCollection loadSents(String name, File comm, String tool) throws IOException {
//        AnnoSentenceReaderPrm prm = new AnnoSentenceReaderPrm();
//        prm.name = name;
//        prm.rePrm.depParseTool = null;
//        prm.rePrm.srlTool = tool;
//        prm.rePrm.sprlTool = tool;
//        // prm.maxNumSentences = trainMaxNumSentences;
//        // prm.maxSentenceLength = trainMaxSentenceLength;
//        // prm.minSentenceLength = trainMinSentenceLength;
//        // prm.useCoNLLXPhead = trainUseCoNLLXPhead;
//        // prm.normalizeRoleNames = normalizeRoleNames;
//        // prm.useGoldSyntax = useGoldSyntax;
//        AnnoSentenceReader reader = new AnnoSentenceReader(prm);
//        reader.loadSents(comm, DatasetType.CONCRETE);
//        return reader.getData();
//    }
//
//    private static List<String> propList(AnnoSentence a, Pair<Integer, Integer> pair, List<String> propOrder) {
//        SprlProperties sprl = a.getSprl();
//        List<String> returnList = new ArrayList<>();
//        for (String k : propOrder) {
//            returnList.add(sprl != null ? sprl.get(pair.get1(),  pair.get2(), k) : SprlLabelConverter.nil());
//        }
//        return returnList;
//    }
///*    
//    public static void findSprlExamples(AnnoSentenceCollection gold, AnnoSentenceCollection pred, File outFile, Set<String> nils) {
//        List<String> propOrder = new ArrayList<>(CorpusHandler.getKnownSprlProperties(gold, pred));
//        // I want to find two pred-arg pairs such that they are in two different
//        // sentences, they have the same predicate and the same gold SRL label,
//        // but different predicted sprl and different gold sprl
//        // for each satisfying pair, I can compute the F1 of the sprl for just
//        // that pair and then return the list of these sorted [by first -F1
//        // between gold and pred, then F1 between gold and other gold, then
//        // sentence length]
//        // build a list of Pair<PairF1, Triple<AnnoSentence, PredIndex,
//        // ArgIndex>> o those pairs that satisfy the constraints
//        // first find all pairs that match a particular predicate, argLabel
//        // Map[(predWord, argLabel) -> Map[(sentenceId) ->
//        // list[(pred,arg,goldSprl,predSprl)]]]
//        int nSentences = pred.size();
//        Map<Pair<String, String>, Map<Integer, List<Quadruple<Integer, Pair<Integer, Integer>, List<String>, List<String>>>>> sprlListsByPredAndSrl = new HashMap<>();
//        for (int i = 0; i < nSentences; i++) {
//            AnnoSentence g = gold.get(i);
//            AnnoSentence p = pred.get(i);
//            for (Pair<Integer, Integer> pair : g.getKnownSprlPairs()) {
//                int predIx = pair.get1();
//                int argIx = pair.get2();
//                String predWord = g.getWord(predIx);
//                String argLabel = g.getSrlGraph().getEdge(predIx, argIx).getLabel();
//                Pair<String, String> predAndSrl = new Pair<>(predWord, argLabel);
//                Map<Integer, List<Quadruple<Integer, Pair<Integer, Integer>, List<String>, List<String>>>> sentToPairs = sprlListsByPredAndSrl
//                        .get(predAndSrl);
//                if (sentToPairs == null) {
//                    sentToPairs = new HashMap<>();
//                    sprlListsByPredAndSrl.put(predAndSrl, sentToPairs);
//                }
//                List<Quadruple<Integer, Pair<Integer, Integer>, List<String>, List<String>>> pairs = sentToPairs
//                        .get(i);
//                if (pairs == null) {
//                    pairs = new LinkedList<>();
//                    sentToPairs.put(i, pairs);
//                }
//                pairs.add(new Quadruple<>(i, pair, propList(g, pair, propOrder), propList(p, pair, propOrder)));
//            }
//        }
//
//        // score, <sentence, pred, arg>
//        List<Triple<Double, Quadruple<Integer, Pair<Integer, Integer>, List<String>, List<String>>, Quadruple<Integer, Pair<Integer, Integer>, List<String>, List<String>>>> scoredExamples = new ArrayList<>();
//        // now collect the scores of each by type
//        for (Pair<String, String> predAndSrl : sprlListsByPredAndSrl.keySet()) {
//            Map<Integer, List<Quadruple<Integer, Pair<Integer, Integer>, List<String>, List<String>>>> sentToPairs = sprlListsByPredAndSrl
//                    .get(predAndSrl);
//            for (Integer sent1Ix : sentToPairs.keySet()) {
//                for (Quadruple<Integer, Pair<Integer, Integer>, List<String>, List<String>> pair1 : sentToPairs
//                        .get(sent1Ix)) {
//                    for (Integer sent2Ix : sentToPairs.keySet()) {
//                        if (sent2Ix <= sent1Ix) {
//                            // skip matches and avoid getting both orders
//                            continue;
//                        }
//                        for (Quadruple<Integer, Pair<Integer, Integer>, List<String>, List<String>> pair2 : sentToPairs
//                                .get(sent2Ix)) {
//                            // only keep if the gold and predicted properties
//                            // don't match
//                            if (pair1.get3().equals(pair2.get3()) || pair1.get4().equals(pair2.get4())) {
//                                continue;
//                            }
//                            // compute the score as the harmonic mean o the two
//                            // F1's
//                            double pair1F1 = getF1(pair1.get3(), pair1.get4(), nils);
//                            double pair2F1 = getF1(pair2.get3(), pair1.get4(), nils);
//                            double score = ConfusionMatrix.harmonicMean(pair1F1, pair2F1);
//                            scoredExamples.add(new Triple<>(score, pair1, pair2));
//                        }
//                    }
//
//                }
//            }
//        }
//        scoredExamples.sort(
//                new Comparator<Triple<Double, Quadruple<Integer, Pair<Integer, Integer>, List<String>, List<String>>, Quadruple<Integer, Pair<Integer, Integer>, List<String>, List<String>>>>() {
//                    @Override
//                    public int compare(
//                            Triple<Double, Quadruple<Integer, Pair<Integer, Integer>, List<String>, List<String>>, Quadruple<Integer, Pair<Integer, Integer>, List<String>, List<String>>> o1,
//                            Triple<Double, Quadruple<Integer, Pair<Integer, Integer>, List<String>, List<String>>, Quadruple<Integer, Pair<Integer, Integer>, List<String>, List<String>>> o2) {
//                        // sort first by F1 between gold and predicted (harmonic
//                        // mean of this across the two instance)
//                        double s1 = o1.get1();
//                        double s2 = o2.get1();
//                        if (s1 != s2) {
//                            return Double.compare(-s1, -s2);
//                        } else {
//                            // then sort by the difference between the two gold
//                            // property vectors
//                            double exampleSim1 = getF1(o1.get2().get3(), o1.get3().get3(), nils);
//                            double exampleSim2 = getF1(o2.get2().get3(), o2.get3().get3(), nils);
//                            if (exampleSim1 != exampleSim2) {
//                                // bigger diff is better
//                                return Double.compare(exampleSim1, exampleSim2);
//                            } else {
//                                // then sort to put shorter examples first
//                                int len1 = gold.get(o1.get2().get1()).size() * gold.get(o1.get3().get1()).size();
//                                int len2 = gold.get(o2.get2().get1()).size() * gold.get(o2.get3().get1()).size();
//                                return Integer.compare(len1, len2);
//                            }
//                        }
//                    }
//                });
//        // write the confusion map to a file
//        try {
//            log.info(String.format("Writing examples to: %s", outFile.getAbsolutePath()));
//            Writer fw = new PrintWriter(outFile);
//
//            int nExamples = 10;
//            int i = 0;
//            for (Triple<Double, Quadruple<Integer, Pair<Integer, Integer>, List<String>, List<String>>, Quadruple<Integer, Pair<Integer, Integer>, List<String>, List<String>>> example : scoredExamples) {
//                Quadruple<Integer, Pair<Integer, Integer>, List<String>, List<String>> ex1 = example
//                        .get2();
//                AnnoSentence s1 = gold.get(ex1.get1());
//                String ex1Str = formatExample(s1, null, ex1.get2(), ex1.get3(), ex1.get4(), propOrder);
//
//                Quadruple<Integer, Pair<Integer, Integer>, List<String>, List<String>> ex2 = example
//                        .get3();
//                AnnoSentence s2 = gold.get(ex2.get1());
//                String ex2Str = formatExample(s2, null, ex2.get2(), ex2.get3(), ex2.get4(), propOrder);
//
//                fw.write(String.format("Example A:\n%s\n\n",
//                        getMarkedSentence(s1, ex1.get2().get1(), ex1.get2().get2(), s1)));
//                fw.write(String.format("Example B:\n%s\n\n",
//                        getMarkedSentence(s2, ex2.get2().get1(), ex2.get2().get2(), s2)));
//                fw.write(String.format("%s\n", hStack(String.format("Example A:\n%s", ex1Str), "     ",
//                        String.format("Example B:\n%s", ex2Str))));
//                fw.write("\n");
//                i++;
//                if (i >= nExamples) {
//                    break;
//                }
//            }
//            fw.close();
//        } catch (FileNotFoundException e) {
//            e.printStackTrace();
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//    }
//*/
//
//
//
//    private static Span getSpan(int head, AnnoSentence goldDepTree) {
//        List<Integer> toks = goldDepTree == null ? Collections.singletonList(head) : goldDepTree.getDescendents(head);
//        // end is the last token index plus one
//        Span span = new Span(toks.get(0), toks.get(toks.size() - 1) + 1);
//        if (span.size() != toks.size()) {
//            log.warn(String.format("marked span includes non-descendant tokens. toks: %s, span: %s", toks.toString(),
//                    span.toString()));
//        }
//        return span;
//    }
//
//    /**
//     * Returns a string representation of the sentence with the boldHead marked
//     * and the emphHeadMarked with its descendents in the goldDepTree if given
//     * (if goldDepTree is null then only the heads are marked)
//     */
//    private static String getMarkedSentence(AnnoSentence s, int boldHead, int emphHead, AnnoSentence goldDepTree) {
//        Span boldSpan = getSpan(boldHead, null);
//        Span emphSpan = getSpan(emphHead, goldDepTree);
//        Span firstSpan;
//        Span secondSpan;
//        String firstMark;
//        String secondMark;
//        String boldMark = "***";
//        String emphMark = "===";
//        if (boldHead < emphHead) {
//            firstSpan = boldSpan;
//            firstMark = boldMark;
//            secondSpan = emphSpan;
//            secondMark = emphMark;
//        } else {
//            firstSpan = emphSpan;
//            firstMark = emphMark;
//            secondSpan = boldSpan;
//            secondMark = boldMark;
//        }
//
//        return String.join(" ", s.getWordsStr(new Span(0, firstSpan.start())),
//                firstMark + s.getWordsStr(firstSpan) + firstMark,
//                s.getWordsStr(new Span(firstSpan.end(), secondSpan.start())),
//                secondMark + s.getWordsStr(secondSpan) + secondMark,
//                s.getWordsStr(new Span(secondSpan.end(), s.size())));
//    }
//
//    /**
//     * Creates a formatted string for the given example predicate, argument pair
//     * @param gold gold annotations for the sentence
//     * @param pred predicted annotations for the sentence
//     * @param pair the predicate argument pair
//     * @param goldLabels gold sprl labels for the pair
//     * @param predictedLabels predicated sprl labels for the pair
//     * @param propertyOrder order in which to output the properties (and which to include)
//     */
//    private static String formatExample(AnnoSentence gold, AnnoSentence pred, Pair<Integer, Integer> pair,
//            List<String> goldLabels, List<String> predictedLabels, List<String> propertyOrder) {
//        StringWriter sw = new StringWriter();
//        int predIx = pair.get1();
//        int argIx = pair.get2();
//        sw.write(String.format("Predicate at %s: %s\n", predIx, gold.getWord(predIx)));
//        sw.write(String.format("Argument at %s (%s): %s\n", argIx, gold.getWord(argIx), getSrlLabel(pair, gold)));
//        if (pred != null) {
//            sw.write(String.format("Predicted SRL: %s\n", getSrlLabel(pair, pred)));
//        }
//        if (propertyOrder.size() > 0) {
//            sw.write(String.format("%30s %15s %15s\n", "Property", "Gold", "Predicted"));
//            for (int q = 0; q < propertyOrder.size(); q++) {
//                sw.write(String.format("%30s %15s %15s\n", propertyOrder.get(q), goldLabels.get(q),
//                        predictedLabels.get(q)));
//            }
//        }
//        sw.write("\n");
//        return sw.toString();
//    }
//
//    private static double getF1(List<String> gold, List<String> pred, Set<String> nils) {
//        return getConfusion(gold, pred, nils).f1();
//    }
//
//    private static ConfusionMatrix<String> getConfusion(List<String> gold, List<String> pred, Set<String> nils) {
//        ConfusionMatrix<String> cm = new ConfusionMatrix<>(nils);
//        for (int i = 0; i < gold.size(); i++) {
//            cm.recordPrediction(gold.get(i), pred.get(i));
//        }
//        return cm;
//    }
//
//    public static String getSrlLabel(Pair<Integer, Integer> predArgPair, AnnoSentence s) {
//        SrlGraph srl = s.getSrlGraph();
//        if (srl != null) {
//            if (srl.getKnownSrlPairs().contains(predArgPair)) {
//                SrlEdge edge = srl.getEdge(predArgPair.get1(), predArgPair.get2());
//                return edge.getLabel();
//            }
//        }
//        return srlNil;
//    }
//
//    public static void evalSrl(AnnoSentenceCollection gold, AnnoSentenceCollection pred, Writer fw) throws IOException {
//        Set<String> nils = Collections.singleton(srlNil);
//        ConfusionMatrix<String> cm = new ConfusionMatrix<>(nils);
//        int nSentences = pred.size();
//        for (int i = 0; i < nSentences; i++) {
//            AnnoSentence g = gold.get(i);
//            AnnoSentence p = pred.get(i);
//            for (Pair<Integer, Integer> e : SrlFactorGraphBuilder.getPossibleRolePairs(gold.size(), g.getKnownPreds(),
//                    g.getKnownSrlPairs(), g.getPairsToSkip(), roleStructure, allowPredArgSelfLoops)) {
//                String gL = getSrlLabel(e, g);
//                String pL = getSrlLabel(e, p);
//                // TODO: if this example is shorter, I'd like to replace the longest previous examples
////                String exStr = cm.numExamples(gL, pL) >= numExamples ? null
////                        : String.format("%s\n\n%s\n", getMarkedSentence(g, e.get1(), e.get2(), g), formatExample(g, p,
////                                e, Collections.emptyList(), Collections.emptyList(), Collections.emptyList()));
//                String exStr = String.format("%s\n\n%s\n", getMarkedSentence(g, e.get1(), e.get2(), g), formatExample(g, p,
//                                e, Collections.emptyList(), Collections.emptyList(), Collections.emptyList()));
//                cm.recordPrediction(gL, pL, exStr, numExamples);
//            }
//        }
//
//        // write the confusion map to a file
//        cm.print("SRL", new TreeSet<>(cm.keySet()), fw);
//
//    }
//
//    /**
//     * evaluate the sprl annotations in pred against the gold annotations in gold;
//     */
//    /*
//    public static void evalSprlMulticlass(AnnoSentenceCollection gold, AnnoSentenceCollection pred, Writer fw, Set<String> nils) {
//        List<String> propOrder = new ArrayList<>(CorpusHandler.getKnownSprlProperties(gold, pred));
//        ConfusionMap<String, String> cms = new ConfusionMap<String, String>(nils);
//        int nSentences = pred.size();
//        for (int i = 0; i < nSentences; i++) {
//            AnnoSentence g = gold.get(i);
//            AnnoSentence p = pred.get(i);
//            SprlEvaluator eval = new SprlEvaluator(roleStructure, allowPredArgSelfLoops, nils);
//            List<Triple<Integer, Integer, String>> examples = g.getSprl().getLabeledProperties();
//            List<String> gLabels = eval.getLabels(g, g);
//            List<String> pLabels = eval.getLabels(p, g);
//            assert gLabels.size() == pLabels.size() && gLabels.size() == examples.size();
//            for (Pair<Integer, Integer> ex : eval.getExamplePairs(p, g)) {
//                int predIx = ex.get1();
//                int argIx = ex.get2();
//                if (goldSrlLabel != null && !goldSrlLabel.equals(g.getSrlGraph().getEdge(predIx, argIx).getLabel())) {
//                    continue;
//                }
//                List<String> pML = propList(p, ex, propOrder);
//                List<String> gML = propList(g, ex, propOrder);
//                
////                Properties gML= g.getSprl().get(ex);
//                
//                
////                String gL = String.valueOf(gLabels.get(x));
////                String pL = String.valueOf(pLabels.get(x));
//
//                
////                String exStr = cms.numExamples(gL, pL, q) >= numExamples ? null
////                        : String.format("%s\n\n%s\n", getMarkedSentence(g, predIx, argIx, g),
////                                formatExample(g, null, new Pair<>(predIx, argIx), Collections.singletonList(gL),
////                                        Collections.singletonList(pL), Collections.singletonList(q)));
////                String exStr = String.format("%s\n\n%s\n", getMarkedSentence(g, predIx, argIx, g),
////                                formatExample(g, null, new Pair<>(predIx, argIx), Collections.singletonList(gL),
////                                        Collections.singletonList(pL), Collections.singletonList(q)));
////                cms.recordPrediction(gL, pL, q, exStr, numExamples);
//            }
//        }
//
//        // set the order
//        List<String> labelOrder = new LinkedList<>(Arrays.asList(SprlLabelConverter.LIKELY, SprlLabelConverter.UNKNOWN,
//                SprlLabelConverter.UNLIKELY, SprlLabelConverter.NA, SprlLabelConverter.NOT_AN_ARG));
//
//        // but only include things we saw
//        for (String k : labelOrder) {
//            if (!cms.getTotal().keySet().contains(k)) {
//                labelOrder.remove(k);
//            }
//        }
//
////        cms.print(labelOrder, fw);
//
//
//    }
//    */
//    
//    public static void evalSprl(AnnoSentenceCollection gold, AnnoSentenceCollection pred, Writer fw, Set<String> nils)
//            throws IOException {
//        ConfusionMap<String, String> cms = new ConfusionMap<String, String>(nils);
//        int nSentences = pred.size();
//        for (int i = 0; i < nSentences; i++) {
//            AnnoSentence g = gold.get(i);
//            AnnoSentence p = pred.get(i);
//            SprlEvaluator eval = new SprlEvaluator();
//            List<Triple<Integer, Integer, String>> examples = g.getSprl().getLabeledProperties();
//            List<String> gLabels = eval.getLabels(g, g);
//            List<String> pLabels = eval.getLabels(p, g);
//            assert gLabels.size() == pLabels.size() && gLabels.size() == examples.size();
//            for (int x = 0; x < examples.size(); x++) {
//                Triple<Integer, Integer, String> example = examples.get(x);
//                int predIx = example.get1();
//                int argIx = example.get2();
//                if (goldSrlLabel != null && !goldSrlLabel.equals(g.getSrlGraph().getEdge(predIx, argIx).getLabel())) {
//                    continue;
//                }
//                String gL = String.valueOf(gLabels.get(x));
//                String pL = String.valueOf(pLabels.get(x));
//                String q = example.get3();
////                String exStr = cms.numExamples(gL, pL, q) >= numExamples ? null
////                        : String.format("%s\n\n%s\n", getMarkedSentence(g, predIx, argIx, g),
////                                formatExample(g, null, new Pair<>(predIx, argIx), Collections.singletonList(gL),
////                                        Collections.singletonList(pL), Collections.singletonList(q)));
//                String exStr = String.format("%s\n\n%s\n", getMarkedSentence(g, predIx, argIx, g),
//                                formatExample(g, null, new Pair<>(predIx, argIx), Collections.singletonList(gL),
//                                        Collections.singletonList(pL), Collections.singletonList(q)));
//                cms.recordPrediction(gL, pL, q, exStr, numExamples);
//            }
//        }
//
//        // set the order
//        List<String> labelOrder = new LinkedList<>(Arrays.asList(SprlLabelConverter.LIKELY, SprlLabelConverter.UNKNOWN,
//                SprlLabelConverter.UNLIKELY, SprlLabelConverter.NA, SprlLabelConverter.NOT_AN_ARG));
//
//        // but only include things we saw
//        for (String k : labelOrder) {
//            if (!cms.getTotal().keySet().contains(k)) {
//                labelOrder.remove(k);
//            }
//        }
//
//        cms.print(labelOrder, fw);
//
//    }
//
//    public static void main(String[] args) {
//        int exitCode = 0;
//        ArgParser parser = null;
//
//        try {
//            parser = new ArgParser(SprlConcreteEvaluator.class);
//            parser.registerClass(SrlEvaluator.class);
//            parser.registerClass(SprlConcreteEvaluator.class);
//            parser.parseArgs(args);
//            AnnoSentenceCollection predSents = loadSents("pred", pred, predTool);
//            AnnoSentenceCollection goldSents = loadSents("gold", gold, goldTool);
//
//            log.info(String.format("Writing confusions to: %s", outFile.getAbsolutePath()));
//            Writer fw = new PrintWriter(outFile);
//
//            // SRL confusions
//            if (includeSrl && predSents.someHaveAt(AT.SRL)) {
//                evalSrl(goldSents, predSents, fw);
//                fw.write("\n");
//            }
//
//            // SPRL confusions
//            if (includeSprl && predSents.someHaveAt(AT.SPRL)) {
//                evalSprl(goldSents, predSents, fw, SprlEvaluator.nilLabels);
//                if (examplesOut != null) {
//                    findSprlExamples(goldSents, predSents, examplesOut, SprlEvaluator.nilLabels);
//                }
//            }
//            fw.close();
//        } catch (ParseException e1) {
//            log.error(e1.getMessage());
//            if (parser != null) {
//                parser.printUsage();
//            }
//            exitCode = 1;
//        } catch (Throwable t) {
//            AbstractParallelAnnotator.logThrowable(log, t);
//            exitCode = 1;
//        }
//        System.exit(exitCode);
//    }

  /**
   * Horizontally stack the lines of the two strings
   */
    public static String hStack(String... a) throws IOException {
        List<List<String>> lines = new ArrayList<List<String>>(a.length);
        List<Integer> maxWidths = new ArrayList<Integer>(a.length);
        int totalWidth = 0;
        int maxLines = 0;
        for (String s : a) {
            List<String> sLines = getLines(s);
            lines.add(sLines);
            int maxW = maxWidth(sLines);
            
            // keep track of how wide each bloack is
            maxWidths.add(maxW);
            
            // keep track of the total max width
            totalWidth += maxW;
            
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
    
    private static String getLine(List<String> lines, int i) {
        if (i < lines.size()) {
            return lines.get(i);
        } else {
            return "";
        }
    }

    /**
     * does not include the line terminating characters
     */
    private static List<String> getLines(String s) throws IOException {
        // adapted from
        // http://stackoverflow.com/questions/13464954/how-do-i-split-a-string-by-line-break
        List<String> lines = new ArrayList<String>();
        BufferedReader rdr = new BufferedReader(new StringReader(s));
        for (String line = rdr.readLine(); line != null; line = rdr.readLine()) {
            lines.add(line);
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
    
}
