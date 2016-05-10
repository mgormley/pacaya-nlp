package edu.jhu.nlp.srl;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.CorpusStatistics;
import edu.jhu.nlp.ObsFeTypedFactor;
import edu.jhu.nlp.data.DepGraph;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.data.simple.IntAnnoSentence;
import edu.jhu.nlp.embed.Embeddings;
import edu.jhu.nlp.fcm.FcmFactor;
import edu.jhu.nlp.srl.SrlFeatureExtractor.SrlFeatureExtractorPrm;
import edu.jhu.nlp.srl.SrlWordFeatures.SrlWordFeaturesPrm;
import edu.jhu.pacaya.gm.feat.ObsFeatureConjoiner;
import edu.jhu.pacaya.gm.feat.ObsFeatureExtractor;
import edu.jhu.pacaya.gm.model.ClampFactor;
import edu.jhu.pacaya.gm.model.FactorGraph;
import edu.jhu.pacaya.gm.model.Var;
import edu.jhu.pacaya.gm.model.Var.VarType;
import edu.jhu.pacaya.gm.model.VarConfig;
import edu.jhu.pacaya.gm.model.VarSet;
import edu.jhu.pacaya.util.FeatureNames;
import edu.jhu.pacaya.util.collections.QLists;
import edu.jhu.prim.iter.IntIter;
import edu.jhu.prim.set.IntSet;
import edu.jhu.prim.tuple.Pair;

/**
 * A factor graph builder for SRL.
 * 
 * @author mmitchell
 * @author mgormley
 */
public class SrlFactorGraphBuilder implements Serializable {

    private static final long serialVersionUID = 1L;

    public static final String TEMPLATE_KEY_FOR_UNKNOWN_SENSE = SrlFactorTemplate.SENSE_UNARY + "_" + CorpusStatistics.UNKNOWN_SENSE;
    public static final String TEMPLATE_KEY_FOR_UNKNOWN_SENSE_ROLE = SrlFactorTemplate.SENSE_ROLE_BINARY + "_" + CorpusStatistics.UNKNOWN_SENSE;
    private static final Logger log = LoggerFactory.getLogger(SrlFactorGraphBuilder.class); 

    /**
     * Parameters for the SrlFactorGraph.
     * @author mgormley
     */
    public static class SrlFactorGraphBuilderPrm implements Serializable {
        private static final long serialVersionUID = 1L;
        /** The structure of the Role variables. */
        public RoleStructure roleStructure = RoleStructure.ALL_PAIRS;
        /** The type of the SRL role / sense variables. */
        public VarType srlVarType = VarType.PREDICTED;
        /**
         * Whether the Role variables (if any) that correspond to predicates not
         * marked with a "Y" should be latent, as opposed to predicted
         * variables.
         */
        public boolean makeUnknownPredRolesLatent = true;
        /** Whether to allow a predicate to assign a role to itself. (This should be turned on for English) */
        public boolean allowPredArgSelfLoops = false;
        /** Whether to include unary factors in the model. (Ignored if there are no Link variables.) */
        public boolean unaryFactors = true;
        /** Whether to include factors between the sense and role variables. */
        public boolean binarySenseRoleFactors = false;
        /** Whether to predict the predicate sense. */
        public boolean predictSense = false;
        /** Whether to predict the predicate positions. */
        public boolean predictPredPos = false;
        /** Feature extractor options for SRL. */
        public SrlFeatureExtractorPrm srlFePrm = new SrlFeatureExtractorPrm();
        /** Whether to use FCM factors. */ 
        public boolean fcmFactors = false;
        /** Whether to treat the embeddings as model parameters. */ 
        public boolean fcmFineTuning = false;
        /** FCM word features flags. */ 
        public SrlWordFeaturesPrm fcmWfPrm = new SrlWordFeaturesPrm();
    }

    public enum RoleStructure {
        /** Defines Role variables each of the "known" predicates with all possible arguments. */
        /** Defines Role variables for only known predicate-argument pairs. */
        PAIRS_GIVEN,
        /**
         * Defines Role variables each of the "known" predicates with all
         * possible arguments.
         */
        PREDS_GIVEN,
        /** The N**2 model. */
        ALL_PAIRS,
        /** Do not predict roles. */
        NO_ROLES,
    }
    
    public enum SrlFactorTemplate {
        ROLE_UNARY,
        SENSE_UNARY, 
        SENSE_ROLE_BINARY,
    }
    
    /**
     * Role variable.
     * 
     * @author mgormley
     */
    public static class RoleVar extends Var {
        
        private static final long serialVersionUID = 1L;

        private int parent;
        private int child;
        
        public RoleVar(VarType type, int numStates, String name, List<String> stateNames, int parent, int child) {
            super(type, numStates, name, stateNames);
            this.parent = parent;
            this.child = child;
        }

        public int getParent() {
            return parent;
        }

        public int getChild() {
            return child;
        }
        
        public static String getNilStateName() {
            return "_";
        }
        
        public int getNilState() {
            return getState(getNilStateName());
        }
        
    }
    
    /**
     * Sense variable. 
     * 
     * @author mgormley
     */
    public static class SenseVar extends Var {

        private static final long serialVersionUID = 1L;

        private int parent;
        
        public SenseVar(VarType type, int numStates, String name, List<String> stateNames, int parent) {
            super(type, numStates, name, stateNames);
            this.parent = parent;
        }

        public int getParent() {
            return parent;
        }

    }

    // Parameters for constructing the factor graph.
    private SrlFactorGraphBuilderPrm prm;

    // Cache of the variables for this factor graph. These arrays may contain
    // null for variables we didn't include in the model.
    private RoleVar[][] roleVars;
    private SenseVar[] senseVars;

    // The sentence length.
    private int n;

    // Cached for reuse by the joint factors.
    private ObsFeatureExtractor obsFe;         
    
    public SrlFactorGraphBuilder(SrlFactorGraphBuilderPrm prm) {
        this.prm = prm;
    }

    /**
     * Helper function that returns the possible predicate argument pairs according to the partial annotations in sent
     * if fromSrl, then the known pairs and known preds will be from the srl information in sent, otherwise it will come
     * from the sprl information
     */
    public static List<Pair<Integer, Integer>> getPossibleRolePairs(AnnoSentence sent, RoleStructure rS,
            boolean allowPredArgSelfLoops, boolean fromSrl) {
        IntSet knownPreds = fromSrl ? sent.getKnownPreds() : sent.getKnownSprlPreds(); 
        Set<Pair<Integer, Integer>> knownPairs = fromSrl ? sent.getKnownSrlPairs() : sent.getKnownSprlPairs(); 
        return getPossibleRolePairs(sent.size(), knownPreds, knownPairs, sent.getPairsToSkip(), rS, allowPredArgSelfLoops); 
    }

    /**
     * Returns a list of pairs (i,j) for which a role variable should be built
     * according to the provided roleStructure.
     * 
     * @param isent
     *            Sentence to get pairs for.
     * @param rS
     *            RoleStructure describing which pairs to include.
     * @param allowPredArgSelfLoops
     *            boolean indicating if pairs (i,i) should be included.
     * @return
     */
    public static List<Pair<Integer, Integer>> getPossibleRolePairs(int n, IntSet knownPreds, Collection<Pair<Integer, Integer>> knownPairs, Collection<Pair<Integer, Integer>> pairsToSkip, RoleStructure rS,
            boolean allowPredArgSelfLoops) {
        List<Pair<Integer, Integer>> toReturn = new ArrayList<>();
        if (rS == RoleStructure.PAIRS_GIVEN) {
            toReturn.addAll(knownPairs);
        } else if (rS == RoleStructure.PREDS_GIVEN) {
            // CoNLL-friendly model; preds given
            IntIter iter = knownPreds.iterator();
            while (iter.hasNext()) {
                int i = iter.next();
                for (int j = 0; j < n; j++) {
                    if (i == j && !allowPredArgSelfLoops) {
                        continue;
                    }
                    toReturn.add(new Pair<>(i, j));
                }
            }
        } else if (rS == RoleStructure.ALL_PAIRS) {
            // n**2 model
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (i != j || allowPredArgSelfLoops) {
                        toReturn.add(new Pair<>(i, j));
                    }
                }
            }
        } else if (rS == RoleStructure.NO_ROLES) {
            // No role variables.
        } else {
            throw new IllegalArgumentException("Unsupported model structure: " + rS);
        }
        if (pairsToSkip != null) {
            toReturn.removeAll(pairsToSkip);
        }
        return toReturn;
    }

    /**
     * Adds factors and variables to the given factor graph.
     */
    public void build(IntAnnoSentence isent, CorpusStatistics cs, ObsFeatureConjoiner ofc,
            FactorGraph fg) {
        AnnoSentence sent = isent.getAnnoSentence();                
        List<String> words = sent.getWords();
        List<String> lemmas = sent.getLemmas();
        IntSet knownPreds = sent.getKnownPreds();
        List<String> roleStateNames = cs.roleStateNames;
        Map<String, List<String>> psMap = cs.predSenseListMap;

        // Create feature extractor.
        obsFe = new SrlFeatureExtractor(prm.srlFePrm, isent, cs, ofc);
        
        boolean predsOrPairsGiven = (prm.roleStructure == RoleStructure.PREDS_GIVEN)
                || (prm.roleStructure == RoleStructure.PAIRS_GIVEN);
        // Check for null arguments.
        if (prm.roleStructure == RoleStructure.PREDS_GIVEN && knownPreds == null) {
            throw new IllegalArgumentException("knownPreds must be non-null");
        }
        if (prm.predictSense && lemmas == null) {
            throw new IllegalArgumentException("lemmas must be non-null");
        }
        if (prm.predictSense && psMap == null) {
            throw new IllegalArgumentException("psMap must be non-null");
        }
        if (prm.roleStructure == RoleStructure.PREDS_GIVEN && prm.predictPredPos) {
            throw new IllegalStateException("PREDS_GIVEN assumes that the predicate positions are always observed.");
        }

        this.n = words.size();

        // Create the Role variables.
        roleVars = new RoleVar[n][n];
        for (Pair<Integer, Integer> e : getPossibleRolePairs(n, knownPreds,
                sent.getKnownSrlPairs(), sent.getPairsToSkip(), prm.roleStructure, prm.allowPredArgSelfLoops)) {
            int i = e.get1();
            int j = e.get2();
            roleVars[i][j] = createRoleVar(i, j, knownPreds, roleStateNames);
        }

        // Create the Sense variables.
        senseVars = new SenseVar[n];
        for (int i = 0; i < n; i++) {
            // Only look at the knownPreds if the predicate positions are given.
            if (!prm.predictPredPos && !knownPreds.contains(i)) {
                // Skip non-predicate positions.
                continue;
            }
            if (!prm.predictSense && prm.predictPredPos) {
                // Positions without sense.
                senseVars[i] = createSenseVar(i, CorpusStatistics.PRED_POSITION_STATE_NAMES);
            } else if (prm.predictSense || prm.predictPredPos) {
                // Sense without positions OR Sense and position.
                //
                // Even if we aren't predicting the predicate position, the
                // training data could contain non-gold known predicate positions so we need to
                // include "_" as a possible value for the sense.
                List<String> senseStateNames = psMap.get(lemmas.get(i));
                if (senseStateNames == null) {
                    senseStateNames = QLists.getList(cs.getDefaultSense(sent.getLemma(i), sent.getFeats(i)));
                }
                if (prm.predictPredPos) {
                    // Include the state of "no predicate".
                    senseStateNames = QLists.cons("_", senseStateNames);
                }
                senseVars[i] = createSenseVar(i, senseStateNames);
            } else {
                // Do not add sense variables.
            }
        }

                
        // Add the factors.
        for (int i = -1; i < n; i++) {
            // Get the lemma or UNK if we don't know it.
            String lemmaForTk = null;
            if (i >= 0) {
                if (prm.predictSense && psMap.get(lemmas.get(i)) != null) {
                    // The template key must include the lemma appended, so that
                    // there is a unique set of model parameters for each predicate.
                    lemmaForTk = lemmas.get(i);
                } else {
                    // If we've never seen this predicate, just give it to the (untrained) unknown classifier.
                    lemmaForTk = CorpusStatistics.UNKNOWN_SENSE;
                }
            }
            // Add the unary factors for the sense variables.
            if (i >= 0 && senseVars[i] != null) {
                if (senseVars[i].getNumStates() > 1) {
                    String templateKey = SrlFactorTemplate.SENSE_UNARY + "_" + lemmaForTk;
                    fg.addFactor(new ObsFeTypedFactor(new VarSet(senseVars[i]), SrlFactorTemplate.SENSE_UNARY, templateKey, ofc, obsFe));
                } else {
                    fg.addFactor(new ClampFactor(senseVars[i], 0));
                }
            }
            // Add the role factors.
            for (int j = 0; j < n; j++) {
                if (i != -1) {
                    // Add unary factors on Roles.
                    if (roleVars[i][j] != null) {
                        VarSet vars = new VarSet(roleVars[i][j]);
                        if (prm.unaryFactors) {
                            fg.addFactor(new ObsFeTypedFactor(vars, SrlFactorTemplate.ROLE_UNARY, ofc, obsFe));
                        }
                        if (prm.fcmFactors) {
                            // HACK: Does this work correctly? We do the same in RelationsFactorGraphBuilder.
                            final FeatureNames alphabet = ofc.fcmAlphabet;
                            Embeddings embeddings = (Embeddings)ofc.embeddings;
                            SrlWordFeatures wf = new SrlWordFeatures(prm.fcmWfPrm, sent, alphabet);
                            fg.addFactor(new FcmFactor(vars, sent, embeddings, ofc, prm.fcmFineTuning, wf));
                        }
                    }
                    // Add binary factors between Role and Sense variables.
                    // (Only added if the sense is ambiguous.)
                    if (prm.binarySenseRoleFactors && senseVars[i] != null && roleVars[i][j] != null
                            && senseVars[i].getNumStates() > 1) {
                        String templateKey = SrlFactorTemplate.SENSE_ROLE_BINARY + "_" + lemmaForTk;
                        fg.addFactor(new ObsFeTypedFactor(new VarSet(senseVars[i], roleVars[i][j]), SrlFactorTemplate.SENSE_ROLE_BINARY, templateKey, ofc, obsFe));
                    }
                }
            }
        }
    }

    // ----------------- Creating Variables -----------------

    private RoleVar createRoleVar(int parent, int child, IntSet knownPreds, List<String> roleStateNames) {
        String roleVarName = "Role_" + parent + "_" + child;
        VarType roleVarType = prm.srlVarType;
        if (prm.makeUnknownPredRolesLatent && !knownPreds.contains((Integer) parent)) {
            roleVarType = VarType.LATENT;
        }
        return new RoleVar(roleVarType, roleStateNames.size(), roleVarName, roleStateNames, parent, child);
    }
    
    private SenseVar createSenseVar(int parent, List<String> senseStateNames) {
        String senseVarName = "Sense_" + parent;
        return new SenseVar(prm.srlVarType, senseStateNames.size(), senseVarName, senseStateNames, parent);            
    }
    
    // ----------------- Public Getters -----------------
    
    /**
     * Gets a Role variable.
     * @param i The parent position.
     * @param j The child position.
     * @return The role variable or null if it doesn't exist.
     */
    public RoleVar getRoleVar(int i, int j) {
        if (0 <= i && i < roleVars.length && 0 <= j && j < roleVars[i].length) {
            return roleVars[i][j];
        } else {
            return null;
        }
    }
    
    /**
     * Gets a predicate Sense variable.
     * @param i The position of the predicate.
     * @return The sense variable or null if it doesn't exist.
     */
    public SenseVar getSenseVar(int i) {
        if (0 <= i && i < senseVars.length) {
            return senseVars[i];
        } else {
            return null;
        }
    }

    public int getSentenceLength() {
        return n;
    }

    public RoleVar[][] getRoleVars() {
        return roleVars;
    }

    public ObsFeatureExtractor getFeatExtractor() {
        return obsFe;
    }

    /* ------------------------- Encode ------------------------- */
    public void addVarAssignments(DepGraph srl, VarConfig vc) {
        for (int i = -1; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i == -1 && senseVars[j] != null && senseVars[j].getType() == VarType.PREDICTED) {
                    if (!prm.predictSense && !prm.predictPredPos) {
                        throw new IllegalStateException("Neither predictSense nor predictPredPos is set. So there shouldn't be any SenseVars.");
                    }
                    SenseVar senseVar = senseVars[j];
                    String predSense = srl.get(i, j);
                    if (predSense != null) { // There is a predicate at token j.
                        if (prm.predictSense) {
                            // Tries to map the sense variable to its label (e.g. argM-TMP).
                            // If the variable state space does not include that label, we
                            // fall back on the UNKNOWN_SENSE constant. If for some reason
                            // the UNKNOWN_SENSE constant isn't present, we just set it to the
                            // first possible sense.
                            if (!tryPut(vc, senseVar, predSense)) {
                                if (!tryPut(vc, senseVar, CorpusStatistics.UNKNOWN_SENSE)) {
                                    // This is a hack to ensure that something is added at test time.
                                    vc.put(senseVar, 0);
                                }
                            }
                        } else { // (prm.predictPredPos && !prm.predictPredSense)
                            // We use CorpusStatistics.UNKNOWN_SENSE to indicate that
                            // there exists a predicate at this position.
                            vc.put(senseVar, CorpusStatistics.UNKNOWN_SENSE);   
                        }
                    } else {
                        // The "_" indicates that there is no predicate at this
                        // position.
                        vc.put(senseVar, "_");
                    }
                }
                if (i != -1 && roleVars[i][j] != null && roleVars[i][j].getType() == VarType.PREDICTED) {
                    RoleVar roleVar = roleVars[i][j];
                    String roleName = srl.get(i, j);
                    if (roleName != null) {
                        int roleNameIdx = roleVar.getState(roleName);
                        // TODO: This isn't quite right...we should really store the actual role name here.
                        if (roleNameIdx == -1) {
                            vc.put(roleVar, CorpusStatistics.UNKNOWN_ROLE);
                        } else {
                            vc.put(roleVar, roleNameIdx);
                        }
                    } else {
                        vc.put(roleVar, "_");
                    }
                }
            }
        }
    }

    /**
     * Trys to put the entry (var, stateName) in vc.
     * @return True iff the entry (var, stateName) was added to vc.
     */
    private static boolean tryPut(VarConfig vc, Var var, String stateName) {
        int stateNameIdx = var.getState(stateName);
        if (stateNameIdx == -1) {
            return false;
        } else {
            vc.put(var, stateName);
            return true;
        }
    }
 
    /* ------------------------- Decode ------------------------- */

    // TODO: We used to decode only the PREDICTED vars, but now decode them all.
    // It's possible this could cause unexpected behavior.
    public DepGraph getSrlGraphFromMbrVarConfig(VarConfig vc) {
        int srlVarCount = 0;
        DepGraph srl = new DepGraph(n);
        for (int i = -1; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i == -1 && senseVars[j] != null) {
                    // Decode the Sense var.
                    SenseVar sense = senseVars[j];
                    if (!vc.contains(sense)) { throw new RuntimeException("VarConfig doesn't contain var: " + sense); }
                    String predSense = vc.getStateName(sense);
                    if ("_".equals(predSense)) {
                        // Predicate ID said there's no predicate here.
                    } else {
                        // Adding the identified predicate.
                        srl.set(i, j, predSense);
                    }
                    srlVarCount++;
                }
                if (i != -1 && roleVars[i][j] != null) {
                    // Decode the Role var.
                    RoleVar role = roleVars[i][j];
                    if (!vc.contains(role)) { throw new RuntimeException("VarConfig doesn't contain var: " + role); }
                    String stateName = vc.getStateName(role);
                    if (!"_".equals(stateName)) {
                        if (srl.get(-1, i) == null) {
                            // We need some predicate sense here.
                            srl.set(-1, i, "NO.SENSE.PREDICTED");
                        }
                        srl.set(i, j, stateName);
                    }
                    srlVarCount++;
                }
            }
        }
        log.trace("SRL var count: {}", srlVarCount);
        return srl;
    }
    
    public SenseVar[] getSenseVars() {
        return senseVars;
    }

    public SrlFactorGraphBuilderPrm getPrm() {
        return prm;
    }

}
