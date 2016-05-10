package edu.jhu.nlp.data.concrete;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.google.common.collect.ImmutableMap;
import edu.jhu.hlt.concrete.AnnotationMetadata;
import edu.jhu.hlt.concrete.Communication;
import edu.jhu.hlt.concrete.Dependency;
import edu.jhu.hlt.concrete.DependencyParse;
import edu.jhu.hlt.concrete.EntityMention;
import edu.jhu.hlt.concrete.EntityMentionSet;
import edu.jhu.hlt.concrete.MentionArgument;
import edu.jhu.hlt.concrete.Property;
import edu.jhu.hlt.concrete.Section;
import edu.jhu.hlt.concrete.Sentence;
import edu.jhu.hlt.concrete.SituationMention;
import edu.jhu.hlt.concrete.SituationMentionSet;
import edu.jhu.hlt.concrete.TaggedToken;
import edu.jhu.hlt.concrete.TokenRefSequence;
import edu.jhu.hlt.concrete.TokenTagging;
import edu.jhu.hlt.concrete.Tokenization;
import edu.jhu.hlt.concrete.UUID;
import edu.jhu.hlt.concrete.serialization.CompactCommunicationSerializer;
import edu.jhu.hlt.concrete.util.ConcreteException;
import edu.jhu.hlt.concrete.uuid.UUIDFactory;
import edu.jhu.nlp.data.NerMention;
import edu.jhu.nlp.data.NerMentions;
import edu.jhu.nlp.data.RelationMention;
import edu.jhu.nlp.data.RelationMentions;
import edu.jhu.nlp.data.Span;
import edu.jhu.nlp.data.conll.SrlGraph;
import edu.jhu.nlp.data.conll.SrlGraph.SrlEdge;
import edu.jhu.nlp.data.conll.SrlGraph.SrlPred;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.data.simple.AnnoSentenceCollection;
import edu.jhu.nlp.features.TemplateLanguage.AT;
import edu.jhu.nlp.sprl.SprlLabelConverter;
import edu.jhu.nlp.sprl.SprlProperties;
import edu.jhu.prim.iter.IntIter;
import edu.jhu.prim.set.IntHashSet;
import edu.jhu.prim.set.IntSet;
import edu.jhu.prim.tuple.Pair;

/**
 * Writer of Concrete files from {@link AnnoSentence}s.
 *
 * @author Travis Wolfe
 * @author mgormley
 */
public class ConcreteWriter {

    public static class ConcreteWriterPrm {
        private static final Logger log = LoggerFactory.getLogger(ConcreteWriterPrm.class);
        /* ----- Whether to include each annotation layer ----- */
        /** Whether to add the dependency parses. */
        public boolean addDepParse = true;
        /** Whether to add SRL. */
        public boolean addSrl = true;
        public boolean addSprl = true;
        /** Whether to add NER mentions. */
        public boolean addNerMentions = true;
        /** Whether to add relations. */
        public boolean addRelations = true;
        /** Whether to add pos tags. */
        public boolean addPos = true;
        /** Whether to add lemmata. */
        public boolean addLemmata = true;
        /* ---------------------------------------------------- */
        /**
         * Whether to write out SRL as a labeled dependency tree (i.e. syntax) or as SituationMentions.
         *
         * If true, we put SRL annotations in as dependency parses.
         * Dependency edges from root (gov=-1) represent predicates,
         * with the edge type giving the predicate sense. Arguments
         * are dependents of their predicate token, with the dependency
         * label capturing the argument label (e.g. "ARG0" and "ARG1").
         *
         * Otherwise, we create a SituationMention for every predicate,
         * which have proper Arguments, each of which includes an EntityMention
         * that is added to its own EntityMentionSet (all EntityMentions created
         * by this tool in a document are unioned before making an EntityMentionSet).
         */
        public boolean srlIsSyntax = false;
        /** Sets the include flag for each annotation type to true, or warns if it's not supported. */
        public void addAnnoTypes(Collection<AT> ats) {
            this.addDepParse = ats.contains(AT.DEP_TREE);
            this.addSrl = ats.contains(AT.SRL);
            this.addSprl = ats.contains(AT.SPRL);
            this.addNerMentions = ats.contains(AT.NER);
            this.addRelations = ats.contains(AT.RELATIONS);

            EnumSet<AT> others = EnumSet.complementOf(EnumSet.of(AT.DEP_TREE, AT.SRL, AT.NER, AT.RELATIONS, AT.SPRL, AT.LEMMA, AT.POS));
            for (AT at : ats) {
                if (others.contains(at)) {
                    log.warn("Annotations of type {} are not supported by ConcreteWriter and will not be added to Concrete Communications.", at);
                }
            }
        }
    }

    private static final Logger log = LoggerFactory.getLogger(ConcreteWriter.class);

    public static final String DEP_PARSE_TOOL = "pacaya-depparse";
    public static final String SRL_TOOL = "pacaya-srl";
    private static final String REL_TOOL = "pacaya-rel";
    public static final String SPRL_TOOL = "pacaya-sprl";
    public static final String SPRL_SRL_TOOL = "pacaya-sprl-srl";
    private static final String NER_TOOL = "pacaya-ner";
    private static final String POS_TOOL = "pos";
    private static final String LEMMA_TOOL = "lemmata";
    private static final String PRED_TYPE = "PREDICATE";
    private final long timestamp;     // time that every annotation that is processed will get
    private final ConcreteWriterPrm prm;

    public ConcreteWriter(ConcreteWriterPrm prm) {
        this.timestamp = System.currentTimeMillis();
        this.prm = prm;
    }

    public void write(AnnoSentenceCollection sents, File out) throws IOException {
        List<Communication> comms = (List<Communication>) sents.getSourceSents();
        if (out.getName().endsWith(".zip")) {
            throw new RuntimeException("Zip file output not yet supported for Concrete");
        } else {
            if (comms.size() == 0) {
                throw new RuntimeException("No Communication in sourceSents field.");
            }
            if (comms.size() > 1) {
                throw new RuntimeException("Multiple Communications in input cannot be written to a single Communication as output.");
            }
            Communication comm = comms.get(0);
            comm = comm.deepCopy();
            addAnnotations(sents, comm);
            try {
                CompactCommunicationSerializer ser = new CompactCommunicationSerializer();
                byte[] bytez =ser.toBytes(comm);
                Files.write(Paths.get(out.getAbsolutePath()), bytez);
            } catch (ConcreteException e) {
                throw new RuntimeException(e);
            }
        }
    }

    /** Adds the annotations from the {@link AnnoSentenceCollection} to the {@link Communication}. */
    public void addAnnotations(AnnoSentenceCollection sents, Communication comm) {
        int numSents = ConcreteUtils.getNumSents(comm);
        if (numSents != sents.size()) {
            log.error(String.format("# sents in Communication = %d # sents in AnnoSentenceCollection = %d", numSents, sents.size()));
            log.error("The number of sentences in the Communication do not match the number in the AnnoSentenceCollection." +
                    "This can occur when the maximum sentence length or the total number of sentences is restricted.");
            //throw new RuntimeException("The number of sentences in the Communication do not match the number in the AnnoSentenceCollection.");
        }
        if (prm.addDepParse) {
            addDependencyParse(sents, comm);
        }
        if (prm.addSrl) {
            addSrlAnnotations(sents, comm, true, false, SRL_TOOL);
        }
        if (prm.addSprl) {
            addSrlAnnotations(sents, comm, false, true, SPRL_TOOL);
        }
        if (prm.addSrl && prm.addSprl) {
            addSrlAnnotations(sents, comm, true, true, SPRL_SRL_TOOL);
        }
        if (prm.addNerMentions || prm.addRelations) {
            addNerMentionsAndRelations(sents, comm);
        }
        if (prm.addLemmata) {
            addLemmata(sents, comm);
        }
        if (prm.addPos) {
            addPos(sents, comm);
        }
    }

    private void addPos(AnnoSentenceCollection sents, Communication comm) {
        if (!sents.someHaveAt(AT.POS)) { return; }
        List<Tokenization> ts = getTokenizationsCorrespondingTo(sents, comm);
        AnnotationMetadata meta = new AnnotationMetadata();
        meta.setTool(POS_TOOL);
        meta.setTimestamp(timestamp);
        for(int i=0; i<sents.size(); i++) {
            Tokenization t = ts.get(i);
            AnnoSentence s = sents.get(i);
            List<TaggedToken> taggedTokens = new ArrayList<>();
            for (int j=0; j < s.size(); j++) {
                TaggedToken taggedToken = new TaggedToken();
                taggedToken.setTag(s.getPosTag(j));
                taggedToken.setTokenIndex(j);
                taggedTokens.add(taggedToken);
            }
            TokenTagging tokenTagging = new TokenTagging(getUUID(), meta, taggedTokens);
            tokenTagging.setTaggingType("POS");
            t.addToTokenTaggingList(tokenTagging);
       }
    }

    private void addLemmata(AnnoSentenceCollection sents, Communication comm) {
        if (!sents.someHaveAt(AT.LEMMA)) { return; }
        List<Tokenization> ts = getTokenizationsCorrespondingTo(sents, comm);
        AnnotationMetadata meta = new AnnotationMetadata();
        meta.setTool(LEMMA_TOOL);
        meta.setTimestamp(timestamp);
        for(int i=0; i<sents.size(); i++) {
            Tokenization t = ts.get(i);
            AnnoSentence s = sents.get(i);
            List<TaggedToken> taggedTokens = new ArrayList<>();
            for (int j=0; j < s.size(); j++) {
                TaggedToken taggedToken = new TaggedToken();
                taggedToken.setTag(s.getLemma(j));
                taggedToken.setTokenIndex(j);
                taggedTokens.add(taggedToken);
            }
            TokenTagging tokenTagging = new TokenTagging(getUUID(), meta, taggedTokens);
            tokenTagging.setTaggingType("LEMMA");
            t.addToTokenTaggingList(tokenTagging);
       }
    }

    /**
     * Adds a dependency parse from each sentence in the {@link AnnoSentenceCollection} to each
     * sentence's concrete.Tokenization.
     */
    public void addDependencyParse(
            AnnoSentenceCollection sents,
            Communication comm) {
        if (!sents.someHaveAt(AT.DEP_TREE)) { return; }
        List<Tokenization> ts = getTokenizationsCorrespondingTo(sents, comm);
        for(int i=0; i<sents.size(); i++) {
            Tokenization t = ts.get(i);
            AnnoSentence s = sents.get(i);
            List<String> depTypes = s.getDeprels();
            int[] parents = s.getParents();
            if (parents != null) {
                t.addToDependencyParseList(makeDepParse(parents, depTypes));
            }
       }
    }

    private DependencyParse makeDepParse(int[] parents, List<String> depRels) {
        if(depRels != null && parents.length != depRels.size()) {
            throw new IllegalArgumentException("Parents length doesn't match depRels length");
        }
        DependencyParse p = new DependencyParse();
        p.setUuid(getUUID());
        AnnotationMetadata meta = new AnnotationMetadata();
        meta.setTool(DEP_PARSE_TOOL);
        meta.setTimestamp(timestamp);
        p.setMetadata(meta);
        p.setDependencyList(new ArrayList<Dependency>());
        for(int i=0; i<parents.length; i++) {
            if (parents[i] == -2) { continue; }
            Dependency d = new Dependency();
            d.setDep(i);
            d.setGov(parents[i]);
            if (depRels != null && depRels.get(i) != null) {
                d.setEdgeType(depRels.get(i));
            }
            p.addToDependencyList(d);
        }
        return p;
    }

    /**
     * behavior depends on {@code this.srlIsSyntax}
     */
    public void addSrlAnnotations(
            AnnoSentenceCollection sents,
            Communication comm, boolean includeSrl, boolean includeSprl, String tool) {
        if (!sents.someHaveAt(AT.SRL) && !sents.someHaveAt(AT.SPRL)) { return; }

        AnnotationMetadata meta = new AnnotationMetadata();
        meta.setTool(tool);
        meta.setTimestamp(timestamp);

        List<Tokenization> tokenizations = getTokenizationsCorrespondingTo(sents, comm);

        if(prm.srlIsSyntax) {
            assert includeSrl;
            // make a dependency parse for every sentence / SRL
            for(int i=0; i<tokenizations.size(); i++) {
                AnnoSentence sent = sents.get(i);
                Tokenization at = tokenizations.get(i);
                if (sent.getSrlGraph() != null) {
                    DependencyParse p = makeDependencyParse(sent.getSrlGraph().toSrlGraph(), sent, meta);
                    at.addToDependencyParseList(p);
                }
            }
        } else {
            // make a SituationMention for every sentence / SRL
            EntityMentionSet ems = new EntityMentionSet();
            ems.setUuid(getUUID());
            ems.setMetadata(meta);
            ems.setMentionList(new ArrayList<EntityMention>());
            SituationMentionSet sms = new SituationMentionSet();
            sms.setUuid(getUUID());
            sms.setMetadata(meta);
            sms.setMentionList(new ArrayList<SituationMention>());
            for(int i=0; i<sents.size(); i++) {
                AnnoSentence sent = sents.get(i);
                Tokenization t = tokenizations.get(i);
                for(SituationMention sm : makeSituationMentions(
                        includeSrl ? sent.getSrlGraph().toSrlGraph() : null,
                        includeSprl ? sent.getSprl() : null, sent.getWords(), t, ems, tool)) {
                    sms.addToMentionList(sm);
                }
            }
            comm.addToEntityMentionSetList(ems);
            comm.addToSituationMentionSetList(sms);
        }
    }

    private DependencyParse makeDependencyParse(SrlGraph srl, AnnoSentence from, AnnotationMetadata meta) {
        DependencyParse p = new DependencyParse();
        p.setUuid(getUUID());
        p.setMetadata(meta);
        p.setDependencyList(new ArrayList<Dependency>());
        for(SrlPred pred : srl.getPreds()) {
            {
                Dependency d = new Dependency();
                d.setGov(-1);
                d.setDep(pred.getPosition());
                d.setEdgeType(pred.getLabel());
                p.addToDependencyList(d);
            }
            for(SrlEdge e : pred.getEdges()) {
                Dependency ed = new Dependency();
                ed.setGov(pred.getPosition());
                ed.setDep(e.getArg().getPosition());
                ed.setEdgeType(e.getLabel());
                p.addToDependencyList(ed);
            }
        }
        return p;
    }

    private List<SituationMention> makeSituationMentions(SrlGraph srl, SprlProperties sprl, List<String> words, Tokenization useUUID, EntityMentionSet addEntityMentionsTo, String tool) {
        List<SituationMention> mentions = new ArrayList<SituationMention>();

        AnnotationMetadata sprlMeta = new AnnotationMetadata();
        sprlMeta.setTool(tool);
        sprlMeta.setTimestamp(timestamp);

    	IntSet combinedPreds = new IntHashSet();
        Set<Pair<Integer, Integer>> combinedPairs = new HashSet<>();

        // add preds and pairs from srl
        if (srl != null) {
    	    combinedPreds.add(srl.getKnownPreds().toNativeArray());
            combinedPairs.addAll(srl.getKnownSrlPairs());
    	}

        SprlLabelConverter labelConverter = null;
        // add preds and pairs from sprl
    	if (sprl != null) {
            combinedPreds.add(sprl.getPreds().toNativeArray());
            combinedPairs.addAll(sprl.getKnownPairs());
            labelConverter = sprl.getConverter();
        }

    	// hold onto the preds so that we can add the arguments
    	SituationMention[] cPreds = new SituationMention[words.size()];

        IntIter predLocItr = combinedPreds.iterator();
    	while (predLocItr.hasNext()) {
                        //for(SrlPred p : srl.getPreds()) {
            int predLoc = predLocItr.next();
    	    SituationMention sm = new SituationMention(getUUID(), new ArrayList<MentionArgument>());
            sm.setSituationType(PRED_TYPE);
            if (srl != null) {
                SrlPred p = srl.getPredAt(predLoc);
                if (p != null) {
                    sm.setSituationKind(p.getLabel());
                }
            }

            // set the text for the predicate
            sm.setText(words.get(predLoc));
            TokenRefSequence smToks = new TokenRefSequence();
            smToks.setAnchorTokenIndex(predLoc);
            // todo: include subtree?
            smToks.setTokenIndexList(Arrays.asList(predLoc));
            smToks.setTokenizationId(useUUID.getUuid());
            sm.setTokens(smToks);
            cPreds[predLoc] = sm;
            mentions.add(sm);
    	}

        for (Pair<Integer, Integer> pair : combinedPairs) {
            MentionArgument a = new MentionArgument();
            int predLoc = pair.get1();
            int argLoc = pair.get2();
            SrlEdge child = (srl == null) ? null : srl.getEdge(pair.get1(), pair.get2());
            if (child != null) {
                a.setRole(child.getLabel());
            } else {
                a.setRole("UNKNOWN");
    	    }
    	    if (sprl != null) {
    	        if (sprl.containsPair(pair)) {
    	            List<Property> props = new ArrayList<>();
    	            for (String property : sprl.getLabeledProperties(pair)) {
    	                String label = sprl.get(predLoc, argLoc, property);
    	                Property newProp = new Property();
    	                newProp.setValue(property);
    	                double response = labelConverter.readProb(label);
    	                if (!labelConverter.readApplicable(label)) {
    	                    response *= -1;
    	                }
    	                newProp.setPolarity(response);
    	                newProp.setMetadata(sprlMeta);
    	                props.add(newProp);
    	            }
    	            a.setPropertyList(props);
                }
    	    }
    	    // make an EntityMention
    	    EntityMention em = new EntityMention();
    	    em.setUuid(getUUID());
    	    em.setEntityType("UNKNOWN");
    	    em.setPhraseType("OTHER");
    	    // set the text for the arg
    	    em.setText(words.get(argLoc));
    	    TokenRefSequence seq = new TokenRefSequence();
    	    em.setTokens(seq);
    	    seq.setAnchorTokenIndex(argLoc);
    	    seq.setTokenIndexList(Arrays.asList(argLoc));
    	    seq.setTokenizationId(useUUID.getUuid());

    	    a.setEntityMentionId(em.getUuid());
    	    addEntityMentionsTo.addToMentionList(em);
    	    cPreds[predLoc].addToArgumentList(a);
        }
        return mentions;
    }

    private void addNerMentionsAndRelations(AnnoSentenceCollection sents, Communication comm) {
        // We require NER if we're adding it or not.
        if (!sents.someHaveAt(AT.NER)) { return; }
        // One map per sentence.
        List<Map<NerMention, EntityMention>> aem2cem = new ArrayList<>();
        if (prm.addNerMentions) {
            // 1.a. If we are adding NerMentions, convert the NerMentions to
            // EntityMentions (storing the below mapping along the
            // way).
            List<EntityMention> cEms = new ArrayList<>();
            List<Tokenization> ts = getTokenizationsCorrespondingTo(sents, comm);
            for(int i=0; i<sents.size(); i++) {
                Tokenization cSent = ts.get(i);
                AnnoSentence aSent = sents.get(i);
                Map<NerMention, EntityMention> a2cForSent = new HashMap<>();
                NerMentions aEms = aSent.getNamedEntities();
                if (aEms != null) {
                    for (NerMention aEm : aEms) {
                        TokenRefSequence cSpan = new TokenRefSequence();
                        cSpan.setTokenIndexList(toIntegerList(aEm.getSpan()));
                        cSpan.setTokenizationId(cSent.getUuid());
                        EntityMention cEm = new EntityMention();
                        cEm.setUuid(getUUID());
                        cEm.setTokens(cSpan);
                        String type = aEm.getEntityType();
                        if (aEm.getEntitySubType() != null) {
                            type += ":" + aEm.getEntitySubType();
                        }
                        cEm.setEntityType(type);
                        if (aEm.getPhraseType() != null) {
                            cEm.setPhraseType(aEm.getPhraseType());
                        }
                        a2cForSent.put(aEm, cEm);
                        cEms.add(cEm);
                    }
                }
                aem2cem.add(a2cForSent);
            }
            AnnotationMetadata cMeta = new AnnotationMetadata();
            cMeta.setTool(NER_TOOL);
            cMeta.setTimestamp(timestamp);
            EntityMentionSet cEmSet = new EntityMentionSet();
            cEmSet.setUuid(getUUID());
            cEmSet.setMetadata(cMeta);
            cEmSet.setMentionList(cEms);
            comm.addToEntityMentionSetList(cEmSet);
        } else {
            assert comm.getEntityMentionSetListSize() == 1;
            // 1.b. Create a mapping from NerMention's to EntityMentions (these will be the existing
            // EntityMentions that we read in.)
            // Guaranteed to exist per above assert.
            EntityMentionSet any = comm.getEntityMentionSetList().get(0);

            ImmutableMap.Builder<UUID, EntityMention> id2cemB = new ImmutableMap.Builder<>();
            any.getMentionList().forEach(e -> id2cemB.put(e.getUuid(), e));
            Map<UUID, EntityMention> id2cem = id2cemB.build();

            for(int i=0; i<sents.size(); i++) {
                Map<NerMention, EntityMention> a2cForSent = new HashMap<>();
                AnnoSentence aSent = sents.get(i);
                for (NerMention aEm : aSent.getNamedEntities()) {
                    EntityMention cEm = id2cem.get(new UUID(aEm.getId()));
                    a2cForSent.put(aEm, cEm);
                }
                aem2cem.add(a2cForSent);
            }
        }

        if (prm.addRelations) {
            if (!sents.someHaveAt(AT.RELATIONS)) { return; }
            // 2. Convert AnnoSentence.getRelations() to Concrete's
            // SituationMentions using the above mapping.
            List<SituationMention> cRels = new ArrayList<>();
            for(int i=0; i<sents.size(); i++) {
                AnnoSentence s = sents.get(i);
                Map<NerMention, EntityMention> a2cForSent = aem2cem.get(i);
                RelationMentions aRels = s.getRelations();
                if (aRels != null) {
                    for (RelationMention aRel : aRels) {
                        List<MentionArgument> cArgs = new ArrayList<>();
                        for (Pair<String, NerMention> aArg : aRel.getNerOrderedArgs()) {
                            MentionArgument cArg = new MentionArgument();
                            cArg.setRole(aArg.get1());
                            cArg.setEntityMentionId(a2cForSent.get(aArg.get2()).getUuid());
                            cArgs.add(cArg);
                        }
                        SituationMention cRel = new SituationMention();
                        cRel.setUuid(getUUID());
                        cRel.setArgumentList(cArgs);
                        String relation = aRel.getType();
                        if (aRel.getSubType() != null) {
                            relation += ":" + aRel.getSubType();
                        }
                        cRel.setSituationKind(relation);
                        cRel.setSituationType("STATE");
                        cRels.add(cRel);
                    }
                }
            }
            AnnotationMetadata cMeta = new AnnotationMetadata();
            cMeta.setTool(REL_TOOL);
            cMeta.setTimestamp(timestamp);
            SituationMentionSet cRelSet = new SituationMentionSet();
            cRelSet.setUuid(getUUID());
            cRelSet.setMetadata(cMeta);
            cRelSet.setMentionList(cRels);
            comm.addToSituationMentionSetList(cRelSet);
        }
    }

    /** Converts a {@link Span} to a list of integers. */
    private static List<Integer> toIntegerList(Span span) {
        List<Integer> ids = new ArrayList<>();
        for (int i=span.start(); i<span.end(); i++) {
            ids.add(i);
        }
        return ids;
    }

    private static List<Tokenization> getTokenizationsCorrespondingTo(AnnoSentenceCollection sentences, Communication from) {
        List<Tokenization> ts = new ArrayList<Tokenization>();
        for(Section s : from.getSectionList()) {
            for(Sentence sent : s.getSentenceList()) {
                ts.add(sent.getTokenization());
            }
        }
        // make sure that the sentences line up
        if(ts.size() != sentences.size()) {
            log.error("Number of sentences don't match");
            //throw new RuntimeException("Number of sentences don't match");
        }
        for(int i=0; i<sentences.size(); i++) {
            if(ts.get(i).getTokenList().getTokenListSize() != sentences.get(i).size()) {
                log.error("Sentence lengths don't match");
                //throw new RuntimeException("Sentence lengths don't match");
            }
        }
        return ts;
    }

    private static UUID getUUID() {
        return UUIDFactory.newUUID();
    }

}
