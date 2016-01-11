package edu.jhu.nlp.sprl;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.CorpusStatistics;
import edu.jhu.nlp.ObsFeTypedFactor;
import edu.jhu.nlp.data.Properties;
import edu.jhu.nlp.data.Properties.Property;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.data.simple.IntAnnoSentence;
import edu.jhu.nlp.features.TemplateLanguage.FeatTemplate;
import edu.jhu.nlp.features.TemplateSets;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.RoleStructure;
import edu.jhu.nlp.srl.SrlFeatureExtractor;
import edu.jhu.nlp.srl.SrlFeatureExtractor.SrlFeatureExtractorPrm;
import edu.jhu.pacaya.gm.feat.ObsFeatureConjoiner;
import edu.jhu.pacaya.gm.model.FactorGraph;
import edu.jhu.pacaya.gm.model.Var;
import edu.jhu.pacaya.gm.model.Var.VarType;
import edu.jhu.pacaya.gm.model.VarConfig;
import edu.jhu.pacaya.gm.model.VarSet;
import edu.jhu.pacaya.util.Prm;
import edu.jhu.prim.tuple.Pair;

public class SprlFactorGraphBuilder {

    private static final Logger log = LoggerFactory.getLogger(SprlFactorGraphBuilder.class);

    public static class SprlFactorGraphBuilderPrm extends Prm {
        private static final long serialVersionUID = 1L;
        /** Feature templates. */
        public List<FeatTemplate> templates = TemplateSets.getFromResource(TemplateSets.naradowskyArgFeatsResource);
        /**
         * The value of the mod for use in the feature hashing trick. If <= 0,
         * feature-hashing will be disabled.
         */
        public int featureHashMod = 1000000;
        /** Whether to include unary factors on each sprl var. */
        public boolean unaryFactors = true;
        /**
         * Whether to include pairwise factors between all sprl vars for a given
         * pred-arg pair.
         */
        public boolean pairwiseFactors = false;
        /** The structure of the Role variables. */
        public RoleStructure roleStructure = RoleStructure.PREDS_GIVEN;
        /**
         * Whether to allow a predicate to assign a role to itself. (This should
         * be turned on for English)
         */
        public boolean allowPredArgSelfLoops = false;
        /** Feature extractor options for SRL. */
        public SrlFeatureExtractorPrm srlFePrm = new SrlFeatureExtractorPrm();
        /**
         * Whether to enforce that all sprl variables for a particular pred-arg pair agree as to whether it is a pred-arg pair
         */
        public boolean enforceSprlNilAgreement = true;

    }

    public static class SprlVar extends Var {
        private static final long serialVersionUID = -8752337536058872155L;
        private int pred;
        private int arg;

        public SprlVar(VarType type, int numStates, String name, List<String> stateNames, int pred, int arg) {
            super(type, numStates, name, stateNames);
            this.pred = pred;
            this.arg = arg;
        }

        public int getPred() {
            return pred;
        }

        public int getArg() {
            return arg;
        }

    }

    
    public enum SprlFactorType {
        SPRL_UNARY, SPRL_PAIRWISE
    }

    private SprlFactorGraphBuilderPrm prm;
    private SprlVar[][][] sprlVars;

    public SprlFactorGraphBuilder(SprlFactorGraphBuilderPrm prm) {
        this.prm = prm;
    }

    public SprlVar[][][] getSprlVars() {
        return sprlVars;
    }

    private SrlFeatureExtractor obsFe = null;
    
    /**
     * Adds factors and variables to the given factor graph.
     */
    public void build(IntAnnoSentence isent, ObsFeatureConjoiner ofc, FactorGraph fg, CorpusStatistics cs) {
        // Create feature extractor.
        this.obsFe = new SrlFeatureExtractor(prm.srlFePrm, isent, cs, ofc);
        ofc.takeNoteOfFeatureHashMod(prm.featureHashMod);
        // create the variables
        int n = isent.size();
        sprlVars = new SprlVar[n][n][Property.values().length];
        for (Pair<Integer, Integer> e : SrlFactorGraphBuilder.getPossibleRolePairs(isent.getAnnoSentence(),
                prm.roleStructure, prm.allowPredArgSelfLoops)) {
            int i = e.get1();
            int j = e.get2();
            VarType sprlType = null; // VarType.PREDICTED;
            for (Property q : Property.values()) {
                // 4-way classification
                String name = "sprl_r" + i + "-" + "a" + j + "_" + q;
                SprlVar v = new SprlVar(sprlType, SprlClassLabel.sprlLabels.size(), name, SprlClassLabel.sprlLabels, i,
                        j);

                // add the variable
                fg.addVar(v);

                // keep track of which variable for this slot
                sprlVars[i][j][q.ordinal()] = v;

                // Add unary factors on Roles.
                if (prm.unaryFactors) {
                    VarSet vars = new VarSet(v);
                    // there will be different parameters for each question (and
                    // these will be separate from factors with any other
                    // FactorType)
                    fg.addFactor(new ObsFeTypedFactor(vars, SprlFactorType.SPRL_UNARY, q, ofc, obsFe));
                }
            }
            if (prm.pairwiseFactors) {
                for (Property q1 : Property.values()) {
                    SprlVar v1 = sprlVars[i][j][q1.ordinal()];
                    for (Property q2 : Property.values()) {
                        SprlVar v2 = sprlVars[i][j][q2.ordinal()];
                        Pair<Property, Property> templateKey = new Pair<>(q1, q2);
                        fg.addFactor(new ObsFeTypedFactor(new VarSet(v1, v2), SprlFactorType.SPRL_PAIRWISE, templateKey,
                                ofc, obsFe));
                    }
                }
            }
        }
    }

    public SrlFeatureExtractor getFeatExtractor() {
        return obsFe; 
    }
    
    // decode
    public void configToAnno(VarConfig varConfig, AnnoSentence toAnnotate) {
        Map<Pair<Integer, Integer>, Properties> sprl = new HashMap<>();
        Set<Integer> sprlPreds = new HashSet<>();
        for (Pair<Integer, Integer> e : SrlFactorGraphBuilder.getPossibleRolePairs(toAnnotate, prm.roleStructure,
                prm.allowPredArgSelfLoops)) {
            int pred = e.get1();
            int arg = e.get2();
            Properties props = new Properties();
            boolean notAnArg = varConfig.getState(sprlVars[pred][arg][0]) == SprlClassLabel.NOT_AN_ARG.ordinal();
            for (Property q : Property.values()) {
                // add the variable to the config
                Var sprlVar = sprlVars[pred][arg][q.ordinal()];
                if (notAnArg) {
                    // make sure that they are consistent (joint factor graph builder should have enforced this)
                    if (varConfig.getState(sprlVar) != SprlClassLabel.NOT_AN_ARG.ordinal()) {                    
                        log.error("inconsistent arg labeling by sprl (some said not-arg others said yes-arg)");
                    }
                } else {
                    double response = SprlClassLabel.getResponse(varConfig.getState(sprlVar));
                    props.add(q.name(), response);
                }
            }
            if (!notAnArg) {
                sprl.put(new Pair<>(pred, arg), props);
                sprlPreds.add(pred);
            }
        }
        toAnnotate.setSprl(sprl);
        toAnnotate.setSprlPreds(sprlPreds);
    }

    // encode
    public void annoToConfig(AnnoSentence goldSent, VarConfig addTo) {
        for (Pair<Integer, Integer> e : SrlFactorGraphBuilder.getPossibleRolePairs(goldSent, prm.roleStructure,
                prm.allowPredArgSelfLoops)) {
            int pred = e.get1();
            int arg = e.get2();
            // TODO: there's currently no way to say that the gold says that
            // some of the labels are NOT possible (this would give more
            // negative evidence)
            Properties props = goldSent.getSprl().get(new Pair<>(pred, arg));
            double responses[] = props != null ? props.toArray() : null;
            // TODO: we are assuming that if SPRL is missing but the pred is a
            // known pred, then this is not an arg

            // only extract variables for sprl if the pred is one that has been
            // annotated with sprl
            if (goldSent.getSprlPreds().contains(pred)) {
                boolean isAnArg = goldSent.getKnownSrlPairs().contains(e);
                for (Property q : Property.values()) {
                    // add the variable to the config
                    Var sprlVar = sprlVars[pred][arg][q.ordinal()];
                    SprlClassLabel label = isAnArg ? SprlClassLabel.getLabel(responses[q.ordinal()])
                            : SprlClassLabel.NOT_AN_ARG;
                    addTo.put(sprlVar, label.ordinal());
                }
            }
        }
    }

}
