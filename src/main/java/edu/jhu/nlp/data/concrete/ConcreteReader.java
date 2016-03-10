package edu.jhu.nlp.data.concrete;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.hlt.concrete.Communication;
import edu.jhu.hlt.concrete.Constituent;
import edu.jhu.hlt.concrete.Dependency;
import edu.jhu.hlt.concrete.DependencyParse;
import edu.jhu.hlt.concrete.EntityMention;
import edu.jhu.hlt.concrete.EntityMentionSet;
import edu.jhu.hlt.concrete.MentionArgument;
import edu.jhu.hlt.concrete.Parse;
import edu.jhu.hlt.concrete.Property;
import edu.jhu.hlt.concrete.Section;
import edu.jhu.hlt.concrete.Sentence;
import edu.jhu.hlt.concrete.SituationMention;
import edu.jhu.hlt.concrete.SituationMentionSet;
import edu.jhu.hlt.concrete.TaggedToken;
import edu.jhu.hlt.concrete.Token;
import edu.jhu.hlt.concrete.TokenList;
import edu.jhu.hlt.concrete.TokenRefSequence;
import edu.jhu.hlt.concrete.TokenTagging;
import edu.jhu.hlt.concrete.Tokenization;
import edu.jhu.hlt.concrete.TokenizationKind;
import edu.jhu.hlt.concrete.UUID;
import edu.jhu.hlt.concrete.serialization.CompactCommunicationSerializer;
import edu.jhu.hlt.concrete.util.ConcreteException;
import edu.jhu.hlt.concrete.util.TokenizationUtils.TagTypes;
import edu.jhu.nlp.data.NerMention;
import edu.jhu.nlp.data.NerMentions;
//import edu.jhu.nlp.data.Properties;
import edu.jhu.nlp.data.RelationMention;
import edu.jhu.nlp.data.RelationMentions;
import edu.jhu.nlp.data.Span;
import edu.jhu.nlp.data.conll.SrlGraph;
import edu.jhu.nlp.data.conll.SrlGraph.SrlArg;
import edu.jhu.nlp.data.conll.SrlGraph.SrlEdge;
import edu.jhu.nlp.data.conll.SrlGraph.SrlPred;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.data.simple.AnnoSentenceCollection;
import edu.jhu.nlp.data.simple.CorpusHandler;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.RoleStructure;
import edu.jhu.pacaya.parse.cky.data.NaryTree;
import edu.jhu.pacaya.util.Prm;
import edu.jhu.prim.Primitives.MutableInt;
import edu.jhu.prim.arrays.IntArrays;
import edu.jhu.prim.map.IntIntHashMap;
import edu.jhu.prim.set.IntHashSet;
import edu.jhu.prim.tuple.Pair;
import edu.jhu.prim.util.Lambda.FnO1ToVoid;

/**
 * Reader of Concrete protocol buffer files.
 *
 * @author mgormley
 */
public class ConcreteReader {

    public static class ConcreteReaderPrm extends Prm {
        private static final long serialVersionUID = 1L;
        public String posTool = null;
        public String cposTool = null;
        public String lemmaTool = null;
        public String chunkTool = null;
        public String depParseTool = null;
        public String parseTool = null;
        public String nerTool = null;
        public String relationTool = null;
        public String srlTool = null;
        public String sprlTool = null;
    }

    private static final Logger log = LoggerFactory.getLogger(ConcreteReader.class);

    private CompactCommunicationSerializer ser = new CompactCommunicationSerializer();
    private int numEntityMentions = 0;
    private int numOverlapingMentions = 0;
    private int numSituationMentions = 0;
    private int numSrlPredicates = 0;
    private ConcreteReaderPrm prm;

    public ConcreteReader(ConcreteReaderPrm prm) {
        this.prm = prm;
        if (prm.cposTool == null) {
            log.warn("Using default pos tagging as cpos tagging!");
        }
    }

    /**
     * Determines the type of the path, reads (possibly multiple) Communications from it, and
     * creates AnnoSentences from them.
     */
    public AnnoSentenceCollection sentsFromPath(File inFile) throws IOException {
    	AnnoSentenceCollection sents;
    	if (inFile.isDirectory()) {
    	    sents = sentsFromDir(inFile);
    	} else if (inFile.getName().endsWith(".zip")) {
    	    sents = sentsFromZipFile(inFile);
        } else {
            sents = sentsFromCommFile(inFile);
        }
        log.debug("Num entity mentions: " + numEntityMentions);
        log.debug("Num overlapping entity mentions: " + numOverlapingMentions);
        log.debug("Num situation mentions: " + numSituationMentions);
        log.debug("Num srl predicates: " + numSrlPredicates);
        return sents;
    }

    public AnnoSentenceCollection sentsFromDir(File inDir) throws IOException {
        try {
            List<File> commFiles = edu.jhu.pacaya.util.files.QFiles.getMatchingFiles(inDir, ".+\\.comm$");
            AnnoSentenceCollection annoSents = new AnnoSentenceCollection();
            for (File commFile : commFiles) {
                Communication comm = ser.fromPathString(commFile.getAbsolutePath());
                addSentences(comm, annoSents);
            }
            return annoSents;
        } catch (ConcreteException e) {
            throw new RuntimeException(e);
        }
    }

    public AnnoSentenceCollection sentsFromZipFile(File zipFile) throws IOException {
        try {
            AnnoSentenceCollection annoSents = new AnnoSentenceCollection();
            try (ZipFile zf = new ZipFile(zipFile)) {
                Enumeration<? extends ZipEntry> e = zf.entries();
                while (e.hasMoreElements()) {
                    ZipEntry ze = e.nextElement();
                    log.trace("Reading communication: " + ze.getName());
                    byte[] bytez = toBytes(zf.getInputStream(ze));
                    Communication comm = ser.fromBytes(bytez);
                    addSentences(comm, annoSents);
                }
            }
            return annoSents;
        } catch (ConcreteException e) {
            throw new RuntimeException(e);
        }
    }

    // Adapted from ThriftIO.
    // TODO: Move to Files?
    /** Reads an input stream into a correctly sized array of bytes. */
    private static byte[] toBytes(InputStream input) throws IOException {
        byte[] buffer = new byte[8192];
        int bytesRead;
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        while ((bytesRead = input.read(buffer)) != -1) {
            baos.write(buffer, 0, bytesRead);
        }
        return baos.toByteArray();
    }

    public AnnoSentenceCollection sentsFromCommFile(File concreteFile) throws IOException {
        try {
            Communication communication = ser.fromPathString(concreteFile.getAbsolutePath());
            AnnoSentenceCollection sents = sentsFromComm(communication);
            return sents;
        } catch (ConcreteException e) {
            throw new RuntimeException(e);
        }
    }

    public AnnoSentenceCollection sentsFromCommInputStream(InputStream is) throws IOException {
        try {
            Communication communication = ser.fromInputStream(is);
            AnnoSentenceCollection sents = sentsFromComm(communication);
            return sents;
        } catch (ConcreteException e) {
            throw new RuntimeException(e);
        }
    }

    public AnnoSentenceCollection sentsFromComm(Communication comm) {
        AnnoSentenceCollection annoSents = new AnnoSentenceCollection();
        addSentences(comm, annoSents);
        return annoSents;
    }

    /**
     * Converts each sentence in communication to a {@link AnnoSentence}
     * and adds it to annoSents.
     */
    protected void addSentences(Communication comm, AnnoSentenceCollection aSents) {
        List<AnnoSentence> tmpSents = new ArrayList<>();

        for (Section cSection : comm.getSectionList()) {
            for (Sentence cSent : cSection.getSentenceList()) {
                Tokenization cToks = cSent.getTokenization();
                tmpSents.add(getAnnoSentence(cToks));
            }
        }

        if (comm.getEntityMentionSetListSize() > 0) {
            addNerMentionsFromEntityMentions(comm, tmpSents, prm.nerTool);

            if (comm.getSituationMentionSetListSize() > 0) {
                // relations
                addRelationsFromSituationMentions(comm, tmpSents, prm.relationTool);
                // srl
                addSrlFromSituationMentions(comm, tmpSents, prm.srlTool);
                // sprl
                addSprlFromSituationMentions(comm, tmpSents, prm.sprlTool);
            }
        }

        aSents.addAll(tmpSents);
        // Update source sentences.
        if (aSents.getSourceSents() == null) {
            aSents.setSourceSents(new ArrayList<Communication>());
        }
        log.trace("Adding Communication in sourceSents");
        ((ArrayList<Communication>)aSents.getSourceSents()).add(comm);
    }

    private void addNerMentionsFromEntityMentions(Communication comm, List<AnnoSentence> tmpSents, String nerTool) {
        List<List<NerMention>> allMentions = getNerMentionsFromEntityMentions(comm, nerTool);

        for (int i=0; i<tmpSents.size(); i++) {
            AnnoSentence aSent = tmpSents.get(i);
            List<NerMention> mentions = allMentions.get(i);
            NerMentions ner = new NerMentions(aSent.size(), mentions);
            numEntityMentions += mentions.size();
            numOverlapingMentions += ner.getNumOverlapping();
            aSent.setNamedEntities(ner);
        }

    }

    public static List<List<NerMention>> getNerMentionsFromEntityMentions(Communication comm, String tool) {
        EntityMentionSet cEms = ConcreteUtils.getFirstEntityMentionSetWithName(comm, tool);
        List<Integer> sentenceLengths = getSentenceLengthsFromCommunication(comm);
        List<List<NerMention>> mentions = new ArrayList<>(sentenceLengths.size());
        for (int i = 0; i < sentenceLengths.size(); i++) {
            mentions.add(new ArrayList<>());
        }

        if (cEms == null) {
            return mentions;
        }

        Map<String, Integer> toksUuid2SentIdx = generateTokUuid2SentIdxMap(comm);
        for (EntityMention cEm : cEms.getMentionList()) {
            TokenRefSequence cEmToks = cEm.getTokens();

            //TODO: Matt's orinal here just did span = getSpan(cEmToks)
            Span span = null;
            if (cEmToks.getTokenIndexList().size() == 0) {
                log.warn("entity with no tokens: " + cEm);
                span = new Span(-1, -1);
            } else {
                span = getSpan(cEmToks);
            }

            int sentIdx = toksUuid2SentIdx.get(cEmToks.getTokenizationId().getUuidString());
            String entityType = cEm.getEntityType();
            String entitySubtype = null;
            // TODO: Remove this SemEval-2010 Task 8 specific logic.
            if (entityType != null) {
                if (entityType.startsWith("I-") || entityType.startsWith("B-")) {
                    entityType = entityType.substring(2);
                    entityType = entityType.replace(':', '_');
                }
            }
            // TODO: Remove this ACE 2005-specific logic.
            if (entityType != null && entityType.contains(":")) {
                String[] splits = entityType.split(":");
                entityType = splits[0];
                entitySubtype = splits[1];
            }
            NerMention aEm = new NerMention(
                    span,
                    entityType,
                    entitySubtype,
                    cEm.getPhraseType(),
                    cEmToks.getAnchorTokenIndex(),
                    cEm.getUuid().getUuidString());
            mentions.get(sentIdx).add(aEm);
        }
        return mentions;
    }


    private void addSprlFromSituationMentions(Communication comm, List<AnnoSentence> tmpSents, String tool) {
        int i = 0;
        for (Map<Pair<Integer, Integer>, Properties> sprl : getSrlFromSituationMentions(comm, tool).get2()) {
            AnnoSentence sent = tmpSents.get(i);
            sent.setSprl(sprl);
            IntHashSet sprlPreds = new IntHashSet();
            for (Pair<Integer, Integer> k : sprl.keySet()) {
                sprlPreds.add(k.get1());
            }
            sent.setKnownSprlPreds(sprlPreds);
            sent.setKnownSprlPairs(new HashSet<>(sprl.keySet()));
            i++;
        }
    }


    private void addSrlFromSituationMentions(Communication comm, List<AnnoSentence> tmpSents, String tool) {
        int i = 0;
        for (SrlGraph g : getSrlFromSituationMentions(comm, tool).get1()) {
            AnnoSentence sent = tmpSents.get(i);
            sent.setSrlGraph(g);
            sent.setKnownPredsFromSrlGraph();
            numSrlPredicates += g.getNumPreds();
            sent.setKnownPairsFromSrlGraph();
            if (CorpusHandler.skipMissingLabels) {
                HashSet<Pair<Integer, Integer>> missingLabels = new HashSet<>();
                for (Pair<Integer, Integer> pair : SrlFactorGraphBuilder.getPossibleRolePairs(sent.size(), sent.getKnownPreds(), sent.getKnownSrlPairs(), null, RoleStructure.PAIRS_GIVEN, true)) {
                    SrlEdge e = sent.getSrlGraph().getEdge(pair.get1(), pair.get2());
                    if (e.getLabel().startsWith("*") && e.getLabel().endsWith("*")) {
                        missingLabels.add(pair);
                    }
                }
                sent.setPairsToSkip(missingLabels);
            }
            i++;
        }
    }


    /**
     * Extracts a list of SrlGraphs corresponding to the sentences in the communication comm and annotations of the given tool
     */
    public static Pair<List<SrlGraph>, List<Map<Pair<Integer, Integer>, Properties>>>  getSrlFromSituationMentions(Communication comm, String tool) {
        SituationMentionSet cSms = ConcreteUtils.getFirstSituationMentionSetWithName(comm, tool);
        List<Integer> sentenceLengths = getSentenceLengthsFromCommunication(comm);
        List<SrlGraph> srlGraphs = new ArrayList<>(sentenceLengths.size());
        List<Map<Pair<Integer, Integer>, Properties>> allSprl = new ArrayList<>(sentenceLengths.size());
        for (int i = 0; i < sentenceLengths.size(); i++) {
            srlGraphs.add(new SrlGraph(sentenceLengths.get(i)));
            allSprl.add(new HashMap<>());
        }

        if (cSms != null) {

            Map<String, NerMention> emId2em = getUuid2ArgsMap(comm, tool);
            Map<String, Integer> emId2SentIdx = getUuid2SentIdxMap(comm, tool);

            for (SituationMention cSm : cSms.getMentionList()) {
                List<MentionArgument> args = cSm.getArgumentList();
                if (args.size() < 1) {
                    log.warn("skipping predicate with no args: " + cSm);
                    continue;
                }

                int sentIdx = emId2SentIdx.get(args.get(0).getEntityMentionId().getUuidString());
                SrlGraph srlGraph = srlGraphs.get(sentIdx);
                Map<Pair<Integer, Integer>, Properties> sprl = allSprl.get(sentIdx);

                // add predicate
                int predLoc = Collections.min(cSm.getTokens().getTokenIndexList());
                SrlPred srlPred = null;
                srlPred = srlGraph.getPredAt(predLoc);
                if (srlPred == null) {
                    srlPred = new SrlPred(predLoc, cSm.getSituationKind());
                    srlGraph.addPred(srlPred);
                }

                for (MentionArgument cArg : args) {
                    String role = cArg.getRole();
                    String cEmId = cArg.getEntityMentionId().getUuidString();
                    if (sentIdx != emId2SentIdx.get(cEmId)) {
                        throw new IllegalStateException("pred with args in difference sentences. Orig sent: " + sentIdx + ", cArg: " + cArg.toString());
                    }

                    // add argument
                    int argLoc = emId2em.get(cEmId).getHead();
                    if (argLoc < 0) {
                        // TODO: consider handling this differently
                        // the layout of args into an array indexed by position below, doesn't lend itself to empty arguments
                        log.warn("skipping invisible argument. cArg: " + cArg);
                        continue;
                    }
                    SrlArg srlArg = srlGraph.getArgAt(argLoc);
                    if (srlArg == null) {
                        srlArg = new SrlArg(argLoc);
                        srlGraph.addArg(srlArg);
                    }
                    srlGraph.addEdge(new SrlEdge(srlPred, srlArg, role));

                    // add properties
                    List<Property> propertyList = cArg.getPropertyList();
                    if (propertyList != null && propertyList.size() > 0) {
                        Properties props = new Properties();
                        for (Property prop : cArg.getPropertyList()) {
                            props.add(prop.getValue(), prop.getPolarity());
                        }
                        sprl.put(new Pair<>(predLoc, argLoc), props);
                    }
                }
            }
        }
        return new Pair<>(srlGraphs, allSprl);
    }

    /**
     * Returns a list of sentence lengths corresponding to all of the sentences in the communication
     * @param comm
     * @return
     */
    public static List<Integer> getSentenceLengthsFromCommunication(Communication comm) {
        List<Integer> sentenceLengths = new ArrayList<>();
        for (Section cSection : comm.getSectionList()) {
            for (Sentence cSent : cSection.getSentenceList()) {
                sentenceLengths.add(cSent.getTokenization().getTokenList().getTokenListSize());
            }
        }
        return sentenceLengths;

    }

    private void addRelationsFromSituationMentions(Communication comm, List<AnnoSentence> tmpSents, String tool) {
        SituationMentionSet cSms = ConcreteUtils.getFirstSituationMentionSetWithName(comm, tool);
        if (cSms == null) {
            return;
        }

        for (int i=0; i<tmpSents.size(); i++) {
            tmpSents.get(i).setRelations(new RelationMentions());
        }

        Map<String, NerMention> emId2em = getUuid2ArgsMap(tmpSents);
        Map<String, Integer> emId2SentIdx = getUuid2SentIdxMap(tmpSents);

        for (SituationMention cSm : cSms.getMentionList()) {
            // Type / subtype.
            String type = cSm.getSituationKind();
            if (type == null) { // For SemEval data.
                type = cSm.getSituationType();
            }
            String subtype = null;
            if (type.contains(":")) {
                String[] splits = type.split(":");
                type = splits[0];
                subtype = splits[1];
            }

            // Arguments and sentence index.
            List<Pair<String,NerMention>> aArgs = new ArrayList<>();
            int sentIdx = -1;
            for (MentionArgument cArg : cSm.getArgumentList()) {
                String role = cArg.getRole();
                UUID cEmId = cArg.getEntityMentionId();
                NerMention aEm = emId2em.get(cEmId.getUuidString());
                aArgs.add(new Pair<String,NerMention>(role, aEm));

                Integer idxI = emId2SentIdx.get(cEmId.getUuidString());
                if (idxI == null) {
                    throw new IllegalStateException("Could not find entity in NerMentions with ID: " + cEmId.getUuidString());
                }
                int idx = idxI;
                if (sentIdx != -1 && sentIdx != idx) {
                    throw new IllegalStateException("Multiple sentence indices for arguments: " + sentIdx + " " + idx);
                }
                sentIdx = idx;
            }

            // Situation's trigger extent.
            Span trigger = null;
            if (cSm.getTokens() != null) {
                trigger = getSpan(cSm.getTokens());
            }
            if (sentIdx < 0) {
                log.warn("situation mention with no args: " + cSm);
            	continue;
            }
            RelationMention aSm = new RelationMention(type, subtype, aArgs, trigger);
            AnnoSentence aSent = tmpSents.get(sentIdx);
            RelationMentions aRels = aSent.getRelations();
            aRels.add(aSm);
            aSent.setRelations(aRels);
        }
        numSituationMentions += cSms.getMentionList().size();
    }


    /**
     * returns a map from uuid to nerMention for the first given EntityMention tool in the given communication
     */
    public static Map<String, NerMention> getUuid2ArgsMap(Communication comm, String tool) {
        Map<String, NerMention> emId2em = new HashMap<>();
        for (List<NerMention> nerMentions : getNerMentionsFromEntityMentions(comm, tool)) {
            for (NerMention aEm : nerMentions) {
                emId2em.put(aEm.getId(), aEm);
            }
        }
        return emId2em;
    }

    /** Gets a map from UUIDs to our sentence indices. */
    public static Map<String, Integer> getUuid2SentIdxMap(Communication comm, String tool) {
        Map<String, Integer> emId2SentIdx = new HashMap<>();
        int i = 0;
        for (List<NerMention> nerMentions : getNerMentionsFromEntityMentions(comm, tool)) {
            for (NerMention aEm : nerMentions) {
                emId2SentIdx.put(aEm.getId(), i);
            }
            i += 1;
        }
        return emId2SentIdx;
    }


    /** Get a map from UUIDs to our entity mentions. */
    private Map<String, NerMention> getUuid2ArgsMap(List<AnnoSentence> tmpSents) {
        Map<String, NerMention> emId2em = new HashMap<>();
        for (int i=0; i<tmpSents.size(); i++) {
            for (NerMention aEm : tmpSents.get(i).getNamedEntities()) {
                emId2em.put(aEm.getId(), aEm);
            }
        }
        return emId2em;
    }

    /** Gets a map from UUIDs to our sentence indices. */
    private Map<String, Integer> getUuid2SentIdxMap(List<AnnoSentence> tmpSents) {
        Map<String, Integer> emId2SentIdx = new HashMap<>();
        for (int i=0; i<tmpSents.size(); i++) {
            for (NerMention aEm : tmpSents.get(i).getNamedEntities()) {
                emId2SentIdx.put(aEm.getId(), i);
            }
        }
        return emId2SentIdx;
    }

    public static Map<String, Integer> generateTokUuid2SentIdxMap(Communication comm) {
        Map<String,Integer> toksUuid2SentIdx = new HashMap<>();
        int i=0;
        for (Section cSection : comm.getSectionList()) {
            for (Sentence cSent : cSection.getSentenceList()) {
                Tokenization cToks = cSent.getTokenization();
                toksUuid2SentIdx.put(cToks.getUuid().getUuidString(), i++);
            }
        }
        return toksUuid2SentIdx;
    }

    private AnnoSentence getAnnoSentence(Tokenization tokenization) {
        TokenizationKind kind = tokenization.getKind();
        if (kind != TokenizationKind.TOKEN_LIST) {
            throw new IllegalArgumentException("tokens must be of kind TOKEN_LIST: " + kind);
        }

        AnnoSentence as = new AnnoSentence();

        // Words
        List<String> words = new ArrayList<String>();
        TokenList tl = tokenization.getTokenList();
        for (Token tok : tl.getTokenList()) {
            words.add(tok.getText());
        }
        as.setWords(words);

        // POS tags, Lemmas, and Chunks.
        TokenTagging posTags = ConcreteUtils.getFirstXTagsWithName(tokenization, TagTypes.POS.name(), prm.posTool);
        TokenTagging cposTags = ConcreteUtils.getFirstXTagsWithName(tokenization, TagTypes.POS.name(), prm.cposTool);
        TokenTagging lemmas = ConcreteUtils.getFirstXTagsWithName(tokenization, TagTypes.LEMMA.name(), prm.lemmaTool);
        TokenTagging chunks = ConcreteUtils.getFirstXTagsWithName(tokenization, "CHUNK", prm.chunkTool);
        as.setPosTags(getTagging(posTags));
        as.setCposTags(getTagging(cposTags));
        as.setLemmas(getTagging(lemmas));
        as.setChunks(getTagging(chunks));

        // Dependency Parse
        if (tokenization.isSetDependencyParseList()) {
            int numWords = words.size();
            log.trace("Reading dependency parse with name {}", prm.depParseTool);
            DependencyParse depParse = ConcreteUtils.getFirstDependencyParseWithName(tokenization, prm.depParseTool);
            Pair<int[], List<String>> pair = getParentsDeprels(depParse, numWords);
            if (pair != null) {
                as.setParents(pair.get1());
                as.setDeprels(pair.get2());
            }
            /* TODO: deal with getting rid of parents now
             * int[] parents = getParents(depParse, numWords);
            as.setParents(parents);
            String[] deprels = getDeprels(depParse, numWords);
            as.setDeprels(Arrays.asList(deprels));
            */
        }

        // Constituency Parse
        if (tokenization.isSetParseList()) {
            NaryTree tree = getParse(ConcreteUtils.getFirstParseWithName(tokenization, prm.parseTool ));
            as.setNaryTree(tree);
        }

        return as;
    }


    private static NaryTree getParse(Parse parse) {
        IntIntHashMap id2idx = new IntIntHashMap();

        List<Constituent> cs = parse.getConstituentList();
        // Create the node for each constituent.
        NaryTree[] trees = new NaryTree[cs.size()];
        for (int i=0; i<cs.size(); i++) {
            Constituent c = cs.get(i);
            id2idx.put(c.getId(), i);
            Span span = new Span(NaryTree.NOT_INITIALIZED, NaryTree.NOT_INITIALIZED);
            if (c.isSetStart() && c.isSetEnding()) {
                span = new Span(c.getStart(), c.getEnding());
            }
            boolean isLexical = (c.getChildList().size() == 0);
            trees[i] = new NaryTree(c.getTag(), span.start(), span.end(), new ArrayList<NaryTree>(), isLexical);
        }

        // Add the children for each node.
        for (int i=0; i<cs.size(); i++) {
            Constituent c = cs.get(i);
            for (int id : c.getChildList()) {
                int j = id2idx.get(id);
                trees[i].addChild(trees[j]);
            }
        }

        // Find the root.
        NaryTree root = trees[0];
        while (root.getParent() != null) {
            root = root.getParent();
        }

        final MutableInt numNodes = new MutableInt(0);
        root.preOrderTraversal(new FnO1ToVoid<NaryTree>() {
            @Override
            public void call(NaryTree obj) {
                numNodes.v++;
            }
        });

        if (numNodes.v != cs.size()) {
            log.warn(String.format("Not all constituents were included in the tree: expected=%d actual=%d", cs.size(), numNodes.v));
        }

        return root;
    }

    private static List<String> getTagging(TokenTagging tagging) {
        if (tagging == null) {
            return null;
        }
        List<String> tags = new ArrayList<String>();
        for (TaggedToken tok : tagging.getTaggedTokenList()) {
            tags.add(tok.getTag());
        }
        return tags;
    }

    private static Pair<int[],List<String>> getParentsDeprels(DependencyParse dependencyParse, int numWords) {
        if (dependencyParse == null) {
            return null;
        }
        // Parents.
        int[] parents = new int[numWords];
        Arrays.fill(parents, -2);
        // Labels.
        List<String> deprels = new ArrayList<>(numWords);
        for (int i=0; i<numWords; i++) {
            deprels.add(null);
        }
        for (Dependency arc : dependencyParse.getDependencyList()) {
            // Parent.
            int c = arc.getDep();
            if (c < 0) {
                throw new IllegalStateException(String.format("Invalid dep value %d for dependendency tree %s", arc.getDep(), dependencyParse.getUuid()));
            }
            if (parents[c] != -2) {
                throw new IllegalStateException("Multiple parents for token: " + dependencyParse);
            }
            if (!arc.isSetGov()) {
                parents[c] = -1;
            } else {
                parents[c] = arc.getGov();
            }
            // Label.
            deprels.set(c, arc.getEdgeType());
        }
        if (IntArrays.contains(parents, -2)) {
            log.trace("Dependency tree contains token(s) with no head: " + dependencyParse.getUuid());
        }
        return new Pair<int[],List<String>>(parents, deprels);
    }

    private static Span getSpan(TokenRefSequence toks) {
        int start = Collections.min(toks.getTokenIndexList());
        int end = Collections.max(toks.getTokenIndexList()) + 1;
        Span span = new Span(start, end);
        return span;
    }

    /**
     * Reads a file containing a single Commmunication concrete object as bytes,
     * converts it to a AnnoSentence and prints it out in human readable
     * form.
     */
    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("usage: java " + ConcreteReader.class + " <input file>");
            System.exit(1);
        }
        File inputFile = new File(args[0]);
        if (!inputFile.exists()) {
            System.err.println("ERROR: File does not exist: " + inputFile);
            System.exit(1);
        }

        System.out.println("Reading file: " + inputFile);
        ConcreteReader reader = new ConcreteReader(new ConcreteReaderPrm());
        for (AnnoSentence sent : reader.sentsFromPath(inputFile)) {
            System.out.println(sent);
        }
    }

}
