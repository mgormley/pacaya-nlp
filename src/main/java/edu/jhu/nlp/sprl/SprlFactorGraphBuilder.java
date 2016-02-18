package edu.jhu.nlp.sprl;

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.CorpusStatistics;
import edu.jhu.nlp.ObsFeTypedFactor;
import edu.jhu.nlp.ObsFeTypedFactorWithNilAgreement;
import edu.jhu.nlp.data.Properties;
import edu.jhu.nlp.data.Properties.Property;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.data.simple.IntAnnoSentence;
import edu.jhu.nlp.features.TemplateLanguage.FeatTemplate;
import edu.jhu.nlp.features.TemplateSets;
import edu.jhu.nlp.joint.JointNlpFactorGraph.IsArgLabel;
import edu.jhu.nlp.joint.JointNlpFactorGraph.JointFactorTemplate;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.RoleStructure;
import edu.jhu.nlp.srl.SrlFeatureExtractor;
import edu.jhu.nlp.srl.SrlFeatureExtractor.SrlFeatureExtractorPrm;
import edu.jhu.pacaya.gm.feat.FeatureVector;
import edu.jhu.pacaya.gm.feat.ObsFeExpFamFactor;
import edu.jhu.pacaya.gm.feat.ObsFeatureConjoiner;
import edu.jhu.pacaya.gm.feat.ObsFeatureExtractor;
import edu.jhu.pacaya.gm.model.FactorGraph;
import edu.jhu.pacaya.gm.model.Var;
import edu.jhu.pacaya.gm.model.Var.VarType;
import edu.jhu.pacaya.gm.model.VarConfig;
import edu.jhu.pacaya.gm.model.VarSet;
import edu.jhu.pacaya.util.Prm;
import edu.jhu.pacaya.util.SerializablePair;
import edu.jhu.prim.set.IntHashSet;
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
        public boolean extraVariablesForNilAgreement = true;

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
    private SrlFeatureExtractor obsFe = null;
    private SprlVar[][][] sprlVars;

    private ObsFeatureExtractor isArgFe = null;
    private Var[][] argVars = null;

    public SprlFactorGraphBuilder(SprlFactorGraphBuilderPrm prm) {
        this.prm = prm;
    }

    public SprlVar[][][] getSprlVars() {
        return sprlVars;
    }

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
        
        // set up isArg feature extractor and a place for the extra variables
        if (prm.extraVariablesForNilAgreement) {
            argVars = new Var[n][n];
            // TODO: replace this with just using the srl feature extractor (we might as well featurize these factors, too)
            isArgFe = new ObsFeatureExtractor() {
                @Override
                public FeatureVector calcObsFeatureVector(ObsFeExpFamFactor factor) {
                    return new FeatureVector();
                }
            };
        }
        
        AnnoSentence asent = isent.getAnnoSentence();
        for (Pair<Integer, Integer> e : SrlFactorGraphBuilder.getPossibleRolePairs(isent.size(),
                asent.getKnownSprlPreds(), asent.getKnownSprlPairs(), prm.roleStructure, prm.allowPredArgSelfLoops)) {
            int i = e.get1();
            int j = e.get2();
            VarType sprlType = VarType.PREDICTED; // TODO: setting this to be null (hoping that inference will figure out what's latent and what isn't causes big problems!)
            if (prm.extraVariablesForNilAgreement) {
                argVars[i][j] = new Var(VarType.PREDICTED, IsArgLabel.values().length, "isarg" + i + "_" + j, IsArgLabel.labels);
            }
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
                if (prm.extraVariablesForNilAgreement) {
                    JointFactorTemplate templateType = JointFactorTemplate.ISARG_SPRL_BINARY; 
                    SerializablePair<JointFactorTemplate, Property> templateKey = new SerializablePair<>(templateType, q);
                    fg.addFactor(new ObsFeTypedFactorWithNilAgreement(Arrays.asList(argVars[i][j], v),
                            Arrays.asList(IsArgLabel.NOT_AN_ARG.ordinal(), SprlClassLabel.NOT_AN_ARG.ordinal()),
                            templateType, templateKey, ofc, isArgFe));
                }
            }
            if (prm.pairwiseFactors) {
                for (Property q1 : Property.values()) {
                    SprlVar v1 = sprlVars[i][j][q1.ordinal()];
                    for (Property q2 : Property.values()) {
                        SprlVar v2 = sprlVars[i][j][q2.ordinal()];
                        Pair<Property, Property> templateKey = new SerializablePair<>(q1, q2);
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
        IntHashSet sprlPreds = new IntHashSet();
        for (Pair<Integer, Integer> e : SrlFactorGraphBuilder.getPossibleRolePairs(toAnnotate.size(),
                toAnnotate.getKnownSprlPreds(), toAnnotate.getKnownSprlPairs(), prm.roleStructure, prm.allowPredArgSelfLoops)) {
            int pred = e.get1();
            int arg = e.get2();
            Properties props = new Properties();
            boolean notAnArg = varConfig.getState(sprlVars[pred][arg][0]) == SprlClassLabel.NOT_AN_ARG.ordinal();
            // we don't need to worry about decoding the argVar since we get that information from the sprl anyway
            for (Property q : Property.values()) {
                // add the variable to the config
                Var sprlVar = sprlVars[pred][arg][q.ordinal()];
                if (notAnArg) {
                    // make sure that they are consistent (joint factor graph builder should have enforced this)
                    if (varConfig.getState(sprlVar) != SprlClassLabel.NOT_AN_ARG.ordinal()) {                    
                        log.debug("inconsistent arg labeling by sprl (some said not-arg others said yes-arg)");
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
        // only overwrite the pairs if they were not given
        if (prm.roleStructure != RoleStructure.PAIRS_GIVEN) {
            toAnnotate.setKnownSprlPairs(new HashSet<>(sprl.keySet()));
        }
        // only overwrite the preds if neither pairs nor preds were given
        if (prm.roleStructure != RoleStructure.PREDS_GIVEN && prm.roleStructure != RoleStructure.PAIRS_GIVEN) {
            toAnnotate.setKnownSprlPreds(sprlPreds);
        }
    }

    // encode
    public void annoToConfig(AnnoSentence goldSent, VarConfig addTo) {
        for (Pair<Integer, Integer> e : SrlFactorGraphBuilder.getPossibleRolePairs(goldSent.size(),
                goldSent.getKnownSprlPreds(), goldSent.getSprl().keySet(), prm.roleStructure, prm.allowPredArgSelfLoops)) {
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
            //if (goldSent.getKnownSprlPreds().contains(pred)) {
            boolean isAnArg = goldSent.getKnownSprlPairs().contains(e);
            if (prm.extraVariablesForNilAgreement) {
                addTo.put(argVars[pred][arg], isAnArg ? IsArgLabel.IS_ARG.ordinal() : IsArgLabel.NOT_AN_ARG.ordinal());
            }
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
