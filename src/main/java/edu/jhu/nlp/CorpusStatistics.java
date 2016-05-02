package edu.jhu.nlp;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.TreeSet;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.data.DepGraph;
import edu.jhu.nlp.data.NerMention;
import edu.jhu.nlp.data.Properties;
import edu.jhu.nlp.data.simple.AlphabetStore;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.relations.RelationMunger;
import edu.jhu.nlp.sprl.SprlClassLabel;
import edu.jhu.pacaya.util.collections.QLists;
import edu.jhu.prim.Primitives.MutableInt;
import edu.jhu.prim.tuple.ComparablePair;
import edu.jhu.prim.tuple.Pair;

/**
 * Extract corpus statistics about a CoNLL-2009 dataset.
 * 
 * @author mmitchell
 * @author mgormley
 */

public class CorpusStatistics implements Serializable {

    /**
     * Parameters for CorpusStatistics.
     */
    public static class CorpusStatisticsPrm implements Serializable {
        private static final long serialVersionUID = 1848012037725581753L;
        // TODO: Remove useGoldSyntax since it's no longer used in CorpusStatistics.
        public boolean useGoldSyntax = false;
        public String language = null;
        /** Cutoff for OOV words. */
        public int cutoff = 3;
        /** Cutoff for topN words. */ 
        public int topN = 800;
    }

    private static final long serialVersionUID = 1L;
    private static final Logger log = LoggerFactory.getLogger(CorpusStatistics.class);
    
    public static final String UNKNOWN_ROLE = "argUNK";
    public static final String UNKNOWN_SENSE = "senseUNK.01";
    public static final List<String> PRED_SENSE_UNK_STATE_NAMES = QLists.getList(UNKNOWN_SENSE);
    public static final List<String> PRED_POSITION_STATE_NAMES = QLists.getList("_", UNKNOWN_SENSE);

    public Set<String> knownWords = new HashSet<String>();
    public Set<String> knownPostags = new HashSet<String>();

    public Set<String> topNWords = new HashSet<String>();
    
    public List<String> linkStateNames;
    public List<String> roleStateNames;
    public List<String> sprlPropertyNames;
    public List<String> sprlStateNames = new ArrayList<>();
    public List<String> relationStateNames;
    public List<String> posTagStateNames;
    // Mapping from predicate form to the set of predicate senses.
    public Map<String,List<String>> predSenseListMap = new HashMap<String,List<String>>();

    public int maxSentLength = 0;

    public AlphabetStore store;
    
    public CorpusStatisticsPrm prm;
    private boolean initialized;
    
    public CorpusStatistics(CorpusStatisticsPrm prm) {
        this.prm = prm;
        initialized = false;
    }

    public void init(Iterable<AnnoSentence> cr) {
        init(cr, true);
    }

    public void init(Iterable<AnnoSentence> cr, boolean initAlphabetStore) {
        if (initAlphabetStore) {
            this.store = new AlphabetStore(cr);
        }      
        
        Map<String,Set<String>> predSenseSetMap = new HashMap<String,Set<String>>();
        Set<String> knownSprlProperties = new HashSet<String>();
        Set<String> knownSprlStates = new HashSet<String>();
        Set<String> knownRoles = new HashSet<String>();
        Set<String> knownLinks = new HashSet<String>();
        Set<String> knownNeTypes = new TreeSet<String>();
        Set<String> knownNeSubtypes = new TreeSet<String>();
        Set<String> knownRelations = new TreeSet<String>();
        Map<String, MutableInt> words = new HashMap<String, MutableInt>();
        initialized = true;
        
        // Store the variable states we have seen before so
        // we know what our vocabulary of possible states are for
        // the Link variable. Applies to knownLinks, knownRoles.
        knownLinks.add("True");
        knownLinks.add("False");
        knownRoles.add(UNKNOWN_ROLE);
        // This is a hack:  '_' won't actually be in any of the defined edges.
        // However, removing this messes up what we assume as default.
        knownRoles.add("_");
        int numTruePosRels = 0;
        int numRels = 0;
        for (AnnoSentence sent : cr) {
            // Need to know max sent length because distance features
            // use these values explicitly; an unknown sentence length in
            // test data will result in an unknown feature.
            if (sent.size() > maxSentLength) {
                maxSentLength = sent.size();
            }
            
            // Word stats.
            for (int position = 0; position < sent.size(); position++) {
                addWord(words, sent.getWord(position));
            }
            
            // POS tag stats.
            if (sent.getPosTags() != null) {
                for (int position = 0; position < sent.size(); position++) {
                    knownPostags.add(sent.getPosTag(position));
                }
            }
            
            // SRL stats.
            DepGraph srl = sent.getSrlGraph();
            if (srl != null) {
                for (int p=-1; p<srl.size(); p++) {
                    for (int c=0; c<srl.size(); c++) {
                        if (srl.get(p, c) != null) {
                            if (p == -1) {
                                // If we don't have lemmas, then this is a map from underscore to all possible
                                // predicate senses.
                                String lemma = (sent.getLemmas() != null) ? sent.getLemma(c) : "_";
                                Set<String> senses = predSenseSetMap.get(lemma);
                                if (senses == null) {
                                    senses = new TreeSet<String>();
                                    predSenseSetMap.put(lemma, senses);
                                }
                                senses.add(srl.get(p, c));
                            } else {
                                knownRoles.add(srl.get(p, c));
                            }
                        }
                    }
                }
            }

            // SPRL stats.
            if (sent.getSprl() != null) {
                for (Map.Entry<Pair<Integer, Integer>, Properties> e : sent.getSprl().entrySet()) {
                    List<String> theseProps = new ArrayList<>(e.getValue().getMap().keySet());
                    // add the properties
                    knownSprlProperties.addAll(theseProps);
                    // and the SPRL class label for each on this pair (e.g. LIKELY, UNLIKLEY)
                    for (SprlClassLabel label : e.getValue().toLabels(theseProps)) {
                        knownSprlStates.add(label.name());
                    }
                }
            }
            
            // Named Entity stats.
            if (sent.getNamedEntities() != null) {
                for (int k=0; k<sent.getNamedEntities().size(); k++) {
                    NerMention ne = sent.getNamedEntities().get(k);
                    if (ne.getEntityType() != null) {
                        knownNeTypes.add(ne.getEntityType());
                        if (ne.getEntitySubType() != null) {
                            knownNeSubtypes.add(ne.getEntityType() + ":" + ne.getEntitySubType());
                        }
                    }
                }
            }
            
            // Relation stats.
            if (sent.getRelLabels() != null) {
            	for (int k=0; k<sent.getRelLabels().size(); k++) {
                    String relation = sent.getRelLabels().get(k);
                    knownRelations.add(relation);
                    if (!RelationMunger.isNoRelationLabel(relation)) {
                    	numTruePosRels++;
                    }
                    numRels++;
                }
            }
        }
        
        // For words and unknown word classes, we only keep those above some threshold.
        knownWords = getUnigramsAboveThreshold(words, prm.cutoff);
        
        topNWords = getTopNUnigrams(words, prm.topN, prm.cutoff);
        
        this.linkStateNames = new ArrayList<>(knownLinks);
        this.roleStateNames =  new ArrayList<>(knownRoles);
        this.sprlPropertyNames =  new ArrayList<>(knownSprlProperties);
        this.sprlStateNames =  new ArrayList<>(knownSprlStates);
        this.relationStateNames =  new ArrayList<>(knownRelations);
        this.posTagStateNames = new ArrayList<>(knownPostags);
        for (Entry<String,Set<String>> entry : predSenseSetMap.entrySet()) {
            predSenseListMap.put(entry.getKey(), new ArrayList<String>(entry.getValue()));
        }

        log.info("Found {} Word types.", words.size());
        log.info("Found {} POS tag types: {}", knownPostags.size(), knownPostags);
        log.info("Found {} SRL Predicate types.", predSenseListMap.size());
        log.info("Found {} SRL Role types: {}", roleStateNames.size(), roleStateNames);
        log.info("Found {} SPRL Properties: {}", sprlPropertyNames.size(), sprlPropertyNames);
        log.info("Found {} SPRL Property State Names: {}", sprlStateNames.size(), sprlStateNames);
        log.info("Found {} NER types: {}", knownNeTypes.size(), knownNeTypes);
        log.info("Found {} NER subtypes: {}", knownNeSubtypes.size(), knownNeSubtypes);
        log.info("Found {} Relation types: {}", relationStateNames.size(), relationStateNames);        
        log.info("Num true positive relations: " + numTruePosRels);
        log.info("Num relations: " + numRels);
    }

    public String getDefaultSense(String lemma, List<String> feats) {
        if ("de".equals(prm.language)) {
            return lemma + ".1";
        } else if ("es".equals(prm.language) || "ca".equals(prm.language)) {
            // This is the linguistically informed default used by Bjorkelund et al. (2009).
            StringBuilder fb = new StringBuilder();
            for (String feat : feats) {
                fb.append(feat);
                fb.append("_");
            }
            String f = fb.toString();
            String suffix;
            if (f.contains("postype=auxiliary")) {
                suffix = "c2";
            } else if (f.contains("postype=common")) {
                suffix = "a2";
            } else if (f.contains("postype=main")) {
                suffix = "a2";
            } else if (f.contains("postype=qualificative")) {
                suffix = "b2";
            } else if (f.contains("postype=semiauxiliary")) {
                suffix = "c2";
            } else {
                suffix = "a2";
            }
            return lemma + "." + suffix;
        } else if ("en".equals(prm.language) || "zh".equals(prm.language)) {
            return lemma + ".01";
        } else {
            throw new RuntimeException("Language has no default predicate sense: " + prm.language);
        }
    }
    
    // ------------------- private ------------------- //
    
    private static void addWord(Map<String, MutableInt> inputHash, String w) {
        MutableInt count = inputHash.get(w);
        if (count == null) {
            inputHash.put(w, new MutableInt(1));
        } else {
            count.v++;
        }
    }


    private static Set<String> getUnigramsAboveThreshold(Map<String, MutableInt> inputHash, int cutoff) {
        Set<String> knownHash = new HashSet<String>();
        Iterator<Entry<String, MutableInt>> it = inputHash.entrySet().iterator();
        while (it.hasNext()) {
            Map.Entry pairs = it.next();
            int count = ((MutableInt) pairs.getValue()).v;
            if (count > cutoff) {
                knownHash.add((String) pairs.getKey());
            }
        }
        return knownHash;
    }

    private static Set<String> getTopNUnigrams(Map<String, MutableInt> map, int topN, int cutoff) {
        List<ComparablePair<Integer, String>> pairs = new ArrayList<ComparablePair<Integer, String>>(map.size());
        for (Entry<String, MutableInt> entry : map.entrySet()) {
            int count = entry.getValue().v;
            if (count > cutoff) {
                pairs.add(new ComparablePair<Integer, String>(count, entry.getKey()));
            }
        }
        Collections.sort(pairs, Collections.reverseOrder());
        HashSet<String> set = new HashSet<String>();
        for (Pair<Integer,String> p : pairs.subList(0, Math.min(pairs.size(), topN))) {
            set.add(p.get2());
        }
        return set;
    }
    
    @Override
    public String toString() {
        return "CorpusStatistics [" 
                + "\n     knownWords=" + knownWords 
                + ",\n     topNWords=" + topNWords
                + ",\n     knownPostags=" + knownPostags 
                + ",\n     linkStateNames=" + linkStateNames
                + ",\n     roleStateNames=" + roleStateNames 
                + ",\n     maxSentLength=" + maxSentLength + "]";
    }

    public boolean isInitialized() {
        return initialized;
    }

    public String getLanguage() {
        return prm.language;
    }

}
