package edu.jhu.nlp.data.simple;

import java.io.File;
import java.io.IOException;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.hlt.concrete.Communication;
import edu.jhu.nlp.data.Properties;
import edu.jhu.nlp.data.concrete.ConcreteUtils;
import edu.jhu.nlp.data.simple.AnnoSentenceReader.AnnoSentenceReaderPrm;
import edu.jhu.nlp.data.simple.AnnoSentenceReader.DatasetType;
import edu.jhu.nlp.data.simple.AnnoSentenceWriter.AnnoSentenceWriterPrm;
import edu.jhu.nlp.depparse.Projectivizer;
import edu.jhu.nlp.features.TemplateLanguage.AT;
import edu.jhu.pacaya.util.cli.Opt;
import edu.jhu.pacaya.util.collections.QSets;
import edu.jhu.prim.sample.Sample;
import edu.jhu.prim.tuple.Pair;

public class CorpusHandler {
    private static final Logger log = LoggerFactory.getLogger(CorpusHandler.class);

    // Options for train data
    @Opt(hasArg = true, description = "Training data input file or directory.")
    public static File train = null;
    @Opt(hasArg = true, description = "Type of training data input.")
    public static DatasetType trainType = null;
    @Opt(hasArg = true, description = "Training data predictions output file.")
    public static File trainPredOut = null;
    @Opt(hasArg = true, description = "Training data gold output file.")
    public static File trainGoldOut = null;
    @Opt(hasArg = true, description = "Type of training data ouput (predicted and gold). null implies same as trainType.")
    public static DatasetType trainTypeOut = null;
    @Opt(hasArg = true, description = "Maximum sentence length for train.")
    public static int trainMaxSentenceLength = Integer.MAX_VALUE;
    @Opt(hasArg = true, description = "Minimum sentence length for train.")
    public static int trainMinSentenceLength = 0;
    @Opt(hasArg = true, description = "Maximum number of sentences to include in train.")
    public static int trainMaxNumSentences = Integer.MAX_VALUE;
    // Options for dev data
    @Opt(hasArg = true, description = "Dev data input file or directory.")
    public static File dev = null;
    @Opt(hasArg = true, description = "Type of dev data.")
    public static DatasetType devType = DatasetType.CONLL_2009;
    @Opt(hasArg = true, description = "Testing data predictions output file.")
    public static File devPredOut = null;
    @Opt(hasArg = true, description = "Dev data gold output file.")
    public static File devGoldOut = null;
    @Opt(hasArg = true, description = "Type of dev data ouput (predicted and gold). null implies same as trainType.")
    public static DatasetType devTypeOut = null;
    @Opt(hasArg = true, description = "Maximum sentence length for dev.")
    public static int devMaxSentenceLength = Integer.MAX_VALUE;
    @Opt(hasArg = true, description = "Maximum number of sentences to include in dev.")
    public static int devMaxNumSentences = Integer.MAX_VALUE;

    // Options for test data
    @Opt(hasArg = true, description = "Test data input file or directory.")
    public static File test = null;
    @Opt(hasArg = true, description = "Type of test data.")
    public static DatasetType testType = DatasetType.CONLL_2009;
    @Opt(hasArg = true, description = "Test data predictions output file.")
    public static File testPredOut = null;
    @Opt(hasArg = true, description = "Test data gold output file.")
    public static File testGoldOut = null;
    @Opt(hasArg = true, description = "Type of test data ouput (predicted and gold). null implies same as trainType.")
    public static DatasetType testTypeOut = null;
    @Opt(hasArg = true, description = "Maximum sentence length for test.")
    public static int testMaxSentenceLength = Integer.MAX_VALUE;
    @Opt(hasArg = true, description = "Maximum number of sentences to include in test.")
    public static int testMaxNumSentences = Integer.MAX_VALUE;
    @Opt(hasArg=true, description="Whether the test data includes the gold labels.")
    public static boolean testHasGold = true;
    @Opt(hasArg = true, description = "Random proportion of train data to allocate as dev data.")
    public static double propTrainAsDev = 0.0;

    // Language-specific options.
    @Opt(hasArg = true, description = "Language identifier (two character language code).")
    public static String language = null;

    // Options for dependency parse pre-processing.
    @Opt(hasArg = true, description = "Whether to projectivize the training depedendency parses")
    public static boolean trainProjectivize = false;

    @Opt(hasArg = true, description = "Whether to create a fresh communication for the communication output (only applies when blahBlahOut == CONCRETE)")
    public static boolean createNewCommunication = false;

    // Options for data munging.
    @Opt(hasArg = true, description = "Whether to use gold POS tags.")
    public static boolean useGoldSyntax = false;
    @Opt(hasArg = true, description = "Whether to normalize the role names (i.e. lowercase and remove themes).")
    public static boolean normalizeRoleNames = false;
    @Opt(hasArg = true, description = "Comma separated list of annotation types for restricting features/data.")
    public static String removeAts = null;
    @Opt(hasArg = true, description = "Comma separated list of annotation types for predicted annotations.")
    public static String predAts = null;
    @Opt(hasArg = true, description = "Comma separated list of annotation types for latent annotations.")
    public static String latAts = null;
    @Opt(hasArg = true, description = "Whether to remove latent annotations.")
    public static boolean removeLatAts = true;
    // Reader-Specific Options
    @Opt(hasArg = true, description = "CoNLL-X: whether to use the P(rojective)HEAD column for parents.")
    public static boolean trainUseCoNLLXPhead = false;
    @Opt(hasArg = true, description = "Tool name of dependency parse for ConcreteReader. (defaults to the first)")
    public static String concreteDepParseTool = null;
    @Opt(hasArg = true, description = "Tool name of SRL for ConcreteReader (will also be used for NER and Relations).")
    public static String concreteSrlTool = null;
    @Opt(hasArg = true, description = "Tool name of SPRL for ConcreteReader.")
    public static String concreteSprlTool = null;
//    @Opt(hasArg = true, description = "Tool name to use when writing dependency parse for ConcreteWriter.")
//    public static String concreteDepParseOutTool = null;
//    @Opt(hasArg = true, description = "Tool name to use when writing SRL for ConcreteWriter (will also be used for NER and Relations and SPRL).")
//    public static String concreteSrlOutTool = null;

    ////// TODO: use these options... /////
    // @Opt(hasArg=true, description="Whether to normalize and clean words.")
    // public static boolean normalizeWords = false;
    ///////////////////////////////////////

    private AnnoSentenceCollection trainGoldSents;
    private AnnoSentenceCollection trainInputSents;
    private AnnoSentenceCollection devGoldSents;
    private AnnoSentenceCollection devInputSents;
    private AnnoSentenceCollection testGoldSents;
    private AnnoSentenceCollection testInputSents;

    private AnnoSentenceCollection trainAsDevSents;

    // -------------------- Train data --------------------------

    public boolean hasTrain() {
        return train != null && trainType != null;
    }

    public AnnoSentenceCollection getTrainGold() throws IOException {
        if (trainGoldSents == null) {
            loadTrain();
        }
        return trainGoldSents;
    }

    public AnnoSentenceCollection getTrainInput() throws IOException {
        if (trainInputSents == null) {
            loadTrain();
        }
        return trainInputSents;
    }

    public void clearTrainCache() {
        trainGoldSents = null;
        trainInputSents = null;
    }

    public void writeTrainPreds(AnnoSentenceCollection trainPredSents) throws IOException {
        writeSents(trainPredOut, trainPredSents, getTrainTypeOut(), "predicted train");
    }

    public DatasetType getTrainTypeOut() {
        return (trainTypeOut != null) ? trainTypeOut : trainType;
    }

    private void loadTrain() throws IOException {
        if (!hasTrain()) {
            return;
        }
        // Read train data.
        AnnoSentenceReaderPrm prm = getDefaultReaderPrm();
        prm.name = "train";
        prm.maxNumSentences = trainMaxNumSentences;
        prm.maxSentenceLength = trainMaxSentenceLength;
        prm.minSentenceLength = trainMinSentenceLength;
        prm.useCoNLLXPhead = trainUseCoNLLXPhead;
        AnnoSentenceReader reader = new AnnoSentenceReader(prm);
        reader.loadSents(train, trainType);

        // Cache gold train data.
        trainGoldSents = reader.getData();
        trainGoldSents = trainGoldSents.getWithAtsRemoved(getRemoveAts());

        if (hasTrain() && propTrainAsDev > 0) {
            // Split into train and dev.
            trainAsDevSents = new AnnoSentenceCollection();
            AnnoSentenceCollection tmp = new AnnoSentenceCollection();
            sample(trainGoldSents, propTrainAsDev, trainAsDevSents, tmp);
            trainGoldSents = tmp;
        }

        // TODO: Maybe move into a pre-processing pipeline.
        if (trainProjectivize) {
            log.info("Projectivizing training trees");
            new Projectivizer().projectivize(trainGoldSents);
        }
        // Cache input train data.
        trainInputSents = trainGoldSents.getWithAtsRemoved(getGoldOnlyAts());
    }

    private static void writeNewConcrete(AnnoSentenceWriter writer, File outfile, AnnoSentenceCollection sents,
            List<AT> ats) throws IOException {
        Object oldSourceSents = sents.getSourceSents();
        Communication comm = ConcreteUtils.ingestText(sents.getText(), "corpus", "corpus", "tokenization");
        sents.setSourceSents(java.util.Collections.singletonList(comm));
        writer.write(outfile, DatasetType.CONCRETE, sents, ats);
        sents.setSourceSents(oldSourceSents);
    }

    private static void writeSents(File outFile, AnnoSentenceCollection sents, DatasetType outType, String writerName)
            throws IOException {
        if (outFile != null) {
            // Write gold train data.
            AnnoSentenceWriterPrm wPrm = new AnnoSentenceWriterPrm();
            wPrm.name = writerName;
            AnnoSentenceWriter writer = new AnnoSentenceWriter(wPrm);
            if (outType == DatasetType.CONCRETE && createNewCommunication) {
                writeNewConcrete(writer, outFile, sents, sents.get(0).getAts());
            } else {
                writer.write(outFile, outType, sents, sents.get(0).getAts()); //new HashSet<AT>());
            }

        }
    }

    public void writeTrainGold() throws IOException {
        writeSents(trainGoldOut, trainGoldSents, getTrainTypeOut(), "gold train");
    }

    /**
     * Splits inList into two other lists.
     *
     * @param inList
     * @param prop
     *            The proportion of inList to sample into outList1.
     * @param outList1
     *            The sample.
     * @param outList2
     *            The remaining (not sampled) entries.
     */
    public static <T> void sample(List<T> inList, double prop, List<T> outList1, List<T> outList2) {
        if (prop < 0 || 1 < prop) {
            throw new IllegalStateException("Invalid proportion: " + prop);
        }
        int numDev = (int) Math.ceil(prop * inList.size());
        log.info("Num train-as-dev examples: " + numDev);
        boolean[] isDev = Sample.sampleWithoutReplacementBooleans(numDev, inList.size());
        for (int i = 0; i < inList.size(); i++) {
            if (isDev[i]) {
                outList1.add(inList.get(i));
            } else {
                outList2.add(inList.get(i));
            }
        }
    }

    // -------------------- Dev data --------------------------

    public boolean hasDev() {
        return (dev != null && devType != null) || (hasTrain() && propTrainAsDev > 0);
    }

    public AnnoSentenceCollection getDevGold() throws IOException {
        if (devGoldSents == null) {
            loadDev();
        }
        return devGoldSents;
    }

    public AnnoSentenceCollection getDevInput() throws IOException {
        if (devInputSents == null) {
            loadDev();
        }
        return devInputSents;
    }

    public void clearDevCache() {
        devGoldSents = null;
        devInputSents = null;
    }

    public void writeDevPreds(AnnoSentenceCollection devPredSents) throws IOException {
        writeSents(devPredOut, devPredSents, getDevTypeOut(), "predicted dev");
    }

    public DatasetType getDevTypeOut() {
        return (devTypeOut != null) ? devTypeOut : devType;
    }

    private void loadDev() throws IOException {
        if (dev != null && devType != null) {
            readDev();
        }
        if (hasTrain() && propTrainAsDev > 0) {
            loadTrainAsDev();
        }
    }

    public void writeDevGold() throws IOException {
        writeSents(devGoldOut, devGoldSents, getDevTypeOut(), "gold dev");
    }

    private void readDev() throws IOException {
        // Read dev data.
        AnnoSentenceReaderPrm prm = getDefaultReaderPrm();
        prm.name = "dev";
        prm.maxNumSentences = devMaxNumSentences;
        prm.maxSentenceLength = devMaxSentenceLength;
        AnnoSentenceReader reader = new AnnoSentenceReader(prm);
        reader.loadSents(dev, devType);

        // Cache gold dev data.
        devGoldSents = reader.getData();
        devGoldSents = devGoldSents.getWithAtsRemoved(getRemoveAts());

        // Cache input dev data.
        devInputSents = devGoldSents.getWithAtsRemoved(getGoldOnlyAts());
    }

    private void loadTrainAsDev() throws IOException {
        if (trainAsDevSents == null) {
            // Ensure that trainAsDevSents is loaded.
            loadTrain();
        }
        if (devGoldSents == null) {
            devGoldSents = new AnnoSentenceCollection();
        }
        for (AnnoSentence sent : trainAsDevSents) {
            devGoldSents.add(sent);
        }
        devInputSents = devGoldSents.getWithAtsRemoved(getGoldOnlyAts());
    }

    // -------------------- Test data --------------------------

    public boolean hasTest() {
        return test != null && testType != null;
    }

    public boolean hasTestGold() {
        return hasTest() && testHasGold;
    }
    public AnnoSentenceCollection getTestGold() throws IOException {
        if (!testHasGold) {
            return null;
        }
        if (testGoldSents == null) {
            loadTest();
        }
        return testGoldSents;
    }

    public AnnoSentenceCollection getTestInput() throws IOException {
        if (testInputSents == null) {
            loadTest();
        }
        return testInputSents;
    }

    public void clearTestCache() {
        testGoldSents = null;
        testInputSents = null;
    }

    public void writeTestPreds(AnnoSentenceCollection testPredSents) throws IOException {
        writeSents(testPredOut, testPredSents, getTestTypeOut(), "predicted test");
    }

    public DatasetType getTestTypeOut() {
        return (testTypeOut != null) ? testTypeOut : testType;
    }

    private void loadTest() throws IOException {
        if (!hasTest()) {
            return;
        }
        // Read test data.
        AnnoSentenceReaderPrm prm = getDefaultReaderPrm();
        prm.name = "test";
        prm.maxNumSentences = testMaxNumSentences;
        prm.maxSentenceLength = testMaxSentenceLength;
        AnnoSentenceReader reader = new AnnoSentenceReader(prm);
        reader.loadSents(test, testType);

        // Cache gold test data.
        testGoldSents = reader.getData();
        testGoldSents = testGoldSents.getWithAtsRemoved(getRemoveAts());
        if (!testHasGold) { testGoldSents = null; }
        // Cache input test data.
        testInputSents = testGoldSents.getWithAtsRemoved(getGoldOnlyAts());
    }

    public void writeTestGold() throws IOException {
        if (!testHasGold) { throw new IllegalStateException("Test does not have gold data to write."); }
        if (testGoldSents != null && testGoldOut != null) {
            // Write gold test data.
            AnnoSentenceWriterPrm wPrm = new AnnoSentenceWriterPrm();
            wPrm.name = "gold test";
            AnnoSentenceWriter writer = new AnnoSentenceWriter(wPrm);
            writer.write(testGoldOut, getTestTypeOut(), testGoldSents, new HashSet<AT>());
        }
    }

    // -------------------- Helper Methods --------------------------
    private String checkedTool(String label, String toolFromParams) {
        if (toolFromParams == null) {
            log.warn(String.format("Since concrete %s tool is null, using first available tool", label));
        } else {
            log.info(String.format("Using concrete %s tool: %s", label, toolFromParams));
        }
        return toolFromParams;
    }

    private AnnoSentenceReaderPrm getDefaultReaderPrm() {
        AnnoSentenceReaderPrm prm = new AnnoSentenceReaderPrm();
        prm.normalizeRoleNames = normalizeRoleNames;
        prm.useGoldSyntax = useGoldSyntax;
        prm.rePrm.depParseTool = checkedTool("depParse", concreteDepParseTool);
        prm.rePrm.srlTool = checkedTool("srl", concreteSrlTool);
        prm.rePrm.sprlTool = checkedTool("sprl", concreteSprlTool);
        return prm;
    }

    /** Gets predicated annotations (included only in the gold data). */
    public static Set<AT> getPredAts() {
        return getAts(predAts);
    }

    /** Gets latent annotations (included only in the gold data). */
    public static Set<AT> getLatAts() {
        return getAts(latAts);
    }

    /** Gets the annotations removed from both gold and input data. */
    public static Set<AT> getRemoveAts() {
        return getAts(removeAts);
    }

    /** Gets predicated and latent annotations (included only in the gold data). */
    public static Set<AT> getPredLatAts() {
        return QSets.union(getPredAts(), getLatAts());
    }

    /** Gets predicated and latent annotations (included only in the gold data). */
    public static Set<AT> getGoldOnlyAts() {
        if (removeLatAts) {
            return QSets.union(getPredAts(), getLatAts());
        } else {
            log.warn("CAUTION: The latent variable annotations have NOT been removed.");
            return getPredAts();
        }
    }

    public static Set<AT> getAts(String atsStr) {
        if (atsStr == null) {
            return Collections.emptySet();
        }
        String[] splits = atsStr.split(",");
        HashSet<AT> ats = new HashSet<>();
        for (String s : splits) {
            ats.add(AT.valueOf(s));
        }
        return ats;
    }

    /** Gets a set containing all the words appearing in train/dev/test. */
    public Set<String> getAllKnownWords() throws IOException {
        log.info("Reading all data to build known words set.");
        Set<String> words = new HashSet<>();
        if (this.hasTrain()) {
            for (AnnoSentence sent : getTrainInput()) {
                words.addAll(sent.getWords());
            }
        }
        if (this.hasDev()) {
            for (AnnoSentence sent : getDevInput()) {
                words.addAll(sent.getWords());
            }
        }
        if (this.hasTest()) {
            for (AnnoSentence sent : getTestInput()) {
                words.addAll(sent.getWords());
            }
        }
        return words;
    }

    public static Set<String> getKnownSprlProperties(AnnoSentenceCollection... data) {
        Set<String> props = new TreeSet<>();
        for (AnnoSentenceCollection collection : data) {
            for (AnnoSentence sent : collection) {
                if (sent.getSprl() != null) {
                    for (Map.Entry<Pair<Integer, Integer>, Properties> e : sent.getSprl().entrySet()) {
                        props.addAll(e.getValue().getMap().keySet());
                    }
                }
            }
        }
        return props;
    }
    
    /** Gets a set containing all the words appearing in train/dev/test. */
    public Set<String> getAllKnownSprlProperties() throws IOException {
        log.info("Reading all data to build known words set.");
        Set<String> props = new TreeSet<>();
        if (this.hasTrain()) {
            props.addAll(getKnownSprlProperties(getTrainInput()));
        }
        if (this.hasDev()) {
            props.addAll(getKnownSprlProperties(getDevInput()));
        }
        if (this.hasTest()) {
            props.addAll(getKnownSprlProperties(getTestInput()));
        }
        return props;
    }
    
}
