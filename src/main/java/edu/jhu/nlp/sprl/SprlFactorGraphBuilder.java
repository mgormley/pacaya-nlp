package edu.jhu.nlp.sprl;

import static edu.jhu.nlp.joint.JointNlpFactorGraph.makeKey;
import static edu.jhu.nlp.joint.JointNlpFactorGraph.JointFactorTemplate.ISARG_SPRL_BINARY;

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
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.data.simple.IntAnnoSentence;
import edu.jhu.nlp.joint.JointNlpFactorGraph.IsArgLabel;
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
import edu.jhu.prim.set.IntHashSet;
import edu.jhu.prim.tuple.Pair;

public class SprlFactorGraphBuilder {

    private static final Logger log = LoggerFactory.getLogger(SprlFactorGraphBuilder.class);

    public static class SprlFactorGraphBuilderPrm extends Prm {
        private static final long serialVersionUID = 1L;
        /** Feature templates. */
        //public List<FeatTemplate> templates = TemplateSets.getFromResource(TemplateSets.naradowskyArgFeatsResource);
        /**
         * The value of the mod for use in the feature hashing trick. If <= 0,
         * feature-hashing will be disabled.
         */
        //public int featureHashMod = 1000000;
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
         * Whether to enforce that all sprl variables for a particular pred-arg
         * pair agree as to whether it is a pred-arg pair
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
    private CorpusStatistics cs = null;
    
    private ObsFeatureExtractor isArgFe = null;
    private Var[][] argVars = null;

    public SprlFactorGraphBuilder(SprlFactorGraphBuilderPrm prm) {
        this.prm = prm;
    }

    public SprlVar[][][] getSprlVars() {
        return sprlVars;
    }

    public void build(IntAnnoSentence isent, ObsFeatureConjoiner ofc, FactorGraph fg, CorpusStatistics cs) {
        build(isent.getAnnoSentence(), ofc, fg, cs, new SrlFeatureExtractor(prm.srlFePrm, isent, cs, ofc));
    }
    
    //TODO: I want record in the sentence pred,arg,property triples should be marginalized over
    /**
     * Adds factors and variables to the given factor graph.
     */
    public void build(AnnoSentence sent, ObsFeatureConjoiner ofc, FactorGraph fg, CorpusStatistics cs,
            SrlFeatureExtractor fe) {
        // Create feature extractor.
        this.obsFe = fe;
        // hold on to the corpus statistics which we will need for decoding
        this.cs = cs;
//        if (ofc != null) {
//             allow for it to be null in the case that no factors are being
//             constructed and it isn't used
//            ofc.takeNoteOfFeatureHashMod(prm.featureHashMod);
//        }

        // create the variables
        int n = sent.size();
        sprlVars = new SprlVar[n][n][cs.sprlPropertyNames.size()];

        // set up isArg feature extractor and a place for the extra variables
        if (prm.extraVariablesForNilAgreement) {
            argVars = new Var[n][n];
            // TODO: replace this with just using the srl feature extractor (we
            // might as well featurize these factors, too)
            isArgFe = new ObsFeatureExtractor() {
                @Override
                public FeatureVector calcObsFeatureVector(ObsFeExpFamFactor factor) {
                    return new FeatureVector();
                }
            };
        }

        for (Pair<Integer, Integer> e : SrlFactorGraphBuilder.getPossibleRolePairs(sent.size(),
                sent.getKnownSprlPreds(), sent.getKnownSprlPairs(), sent.getPairsToSkip(), prm.roleStructure,
                prm.allowPredArgSelfLoops)) {
            int i = e.get1();
            int j = e.get2();
            VarType sprlType = VarType.PREDICTED; // TODO: setting this to be
                                                  // null (hoping that inference
                                                  // will figure out what's
                                                  // latent and what isn't
                                                  // causes big problems!)
            if (prm.extraVariablesForNilAgreement) {
                argVars[i][j] = new Var(VarType.PREDICTED, IsArgLabel.values().length, "isarg" + i + "_" + j,
                        IsArgLabel.labels);
            }
            for (int qix = 0; qix < cs.sprlPropertyNames.size(); qix++) {
                String q = cs.sprlPropertyNames.get(qix);
                // variable for label on property for this pair
                String name = "sprl_r" + i + "-" + "a" + j + "_" + q;
                SprlVar v = new SprlVar(sprlType, SprlClassLabel.getLabels().size(), name, SprlClassLabel.getLabels(),
                        i, j);

                // add the variable
                fg.addVar(v);

                // keep track of which variable for this slot
                sprlVars[i][j][qix] = v;

                // Add unary factors on properties.
                if (prm.unaryFactors) {
                    VarSet vars = new VarSet(v);
                    // there will be different parameters for each question (and
                    // these will be separate from factors with any other
                    // FactorType)
                    fg.addFactor(new ObsFeTypedFactor(vars, SprlFactorType.SPRL_UNARY,
                            makeKey(SprlFactorType.SPRL_UNARY, q), ofc, obsFe));
                }
                if (prm.extraVariablesForNilAgreement) {
                    fg.addFactor(new ObsFeTypedFactorWithNilAgreement(Arrays.asList(argVars[i][j], v),
                            Arrays.asList(IsArgLabel.NOT_AN_ARG.ordinal(), SprlClassLabel.NOT_AN_ARG.ordinal()),
                            ISARG_SPRL_BINARY, makeKey(ISARG_SPRL_BINARY, q), ofc, isArgFe));
                }
            }
            if (prm.pairwiseFactors) {
                for (int qix1 = 0; qix1 < cs.sprlPropertyNames.size(); qix1++) {
                    String q1 = cs.sprlPropertyNames.get(qix1);
                    SprlVar v1 = sprlVars[i][j][qix1];
                    for (int qix2 = 0; qix2 < cs.sprlPropertyNames.size(); qix2++) {
                        String q2 = cs.sprlPropertyNames.get(qix2);
                        SprlVar v2 = sprlVars[i][j][qix2];
                        fg.addFactor(new ObsFeTypedFactor(new VarSet(v1, v2),
                                SprlFactorType.SPRL_PAIRWISE,
                                makeKey(SprlFactorType.SPRL_PAIRWISE, q1, q2),
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
    // read the values of the sprl variables in the var config and add the
    // corresponding sprl info to the sentence
    public void configToAnno(VarConfig varConfig, AnnoSentence toAnnotate) {
        Map<Pair<Integer, Integer>, Properties> sprl = new HashMap<>();
        IntHashSet sprlPreds = new IntHashSet();
        for (Pair<Integer, Integer> e : SrlFactorGraphBuilder.getPossibleRolePairs(toAnnotate.size(),
                toAnnotate.getKnownSprlPreds(), toAnnotate.getKnownSprlPairs(), toAnnotate.getPairsToSkip(),
                prm.roleStructure, prm.allowPredArgSelfLoops)) {
            int pred = e.get1();
            int arg = e.get2();
            Properties props = new Properties();
            // we don't need to worry about decoding the argVar since we get
            // that information from the sprl anyway
            for (int qix = 0; qix < cs.sprlPropertyNames.size(); qix++) {
                String q = cs.sprlPropertyNames.get(qix);
                // add the variable to the config
                Var sprlVar = sprlVars[pred][arg][qix];
                Double response = SprlClassLabel.getResponse(SprlClassLabel.valueOf(varConfig.getStateName(sprlVar)));
                if (response != null) {
                    props.add(q, response);
                }
            }
            if (props.size() > 0) {
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
    // add sprl variables with values as found in the gold sentence
    public void annoToConfig(AnnoSentence goldSent, VarConfig addTo) {
        // we will create some sprl variables possibly looking at the known sprl
        // predicates and pairs
        for (Pair<Integer, Integer> e : SrlFactorGraphBuilder.getPossibleRolePairs(goldSent.size(),
                goldSent.getKnownSprlPreds(), goldSent.getKnownSprlPairs(), goldSent.getPairsToSkip(),
                prm.roleStructure, prm.allowPredArgSelfLoops)) {
            int pred = e.get1();
            int arg = e.get2();
            // TODO: there's currently no way to say that the gold says that
            // some of the labels are NOT possible (this would give more
            // negative evidence)
            Properties props = goldSent.getSprl().get(new Pair<>(pred, arg));
            List<SprlClassLabel> labels = null;
            boolean isAnArg = false;
            if (props != null) {
                isAnArg = true;
                labels = props.toLabels(cs.sprlPropertyNames);
            }

            // TODO: we are assuming that if SPRL is missing but the pred is a
            // known pred, then this is not an arg
            if (prm.extraVariablesForNilAgreement) {
                addTo.put(argVars[pred][arg], isAnArg ? IsArgLabel.IS_ARG.ordinal() : IsArgLabel.NOT_AN_ARG.ordinal());
            }
            for (int qix = 0; qix < cs.sprlPropertyNames.size(); qix++) {
                // add the variable to the config
                Var sprlVar = sprlVars[pred][arg][qix];
                SprlClassLabel label = isAnArg ? labels.get(qix) : SprlClassLabel.NOT_AN_ARG;
                addTo.put(sprlVar, label.name());
            }
        }
    }

}
