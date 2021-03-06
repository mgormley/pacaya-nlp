package edu.jhu.nlp.sprl;

import static edu.jhu.nlp.joint.JointNlpFactorGraph.makeKey;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import edu.jhu.nlp.CorpusStatistics;
import edu.jhu.nlp.ObsFeTypedFactor;
import edu.jhu.nlp.data.conll.SrlGraph.SrlEdge;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.data.simple.IntAnnoSentence;
import edu.jhu.nlp.joint.JointNlpFactorGraph.JointFactorTemplate;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.RoleStructure;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.RoleVar;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.SrlFactorGraphBuilderPrm;
import edu.jhu.nlp.srl.SrlFeatureExtractor;
import edu.jhu.nlp.srl.SrlFeatureExtractor.SrlFeatureExtractorPrm;
import edu.jhu.pacaya.gm.feat.ObsFeatureConjoiner;
import edu.jhu.pacaya.gm.feat.ObsFeatureExtractor;
import edu.jhu.pacaya.gm.model.FactorGraph;
import edu.jhu.pacaya.gm.model.Var;
import edu.jhu.pacaya.gm.model.Var.VarType;
import edu.jhu.pacaya.gm.model.VarConfig;
import edu.jhu.pacaya.gm.model.VarSet;
import edu.jhu.pacaya.sch.util.Indexed;
import edu.jhu.pacaya.util.Prm;
import edu.jhu.prim.tuple.Pair;

public class SprlFactorGraphBuilder {

    // private static final Logger log =
    // LoggerFactory.getLogger(SprlFactorGraphBuilder.class);

    public static class SprlFactorGraphBuilderPrm extends Prm {
        private static final long serialVersionUID = 1L;
        /**
         * The value of the mod for use in the feature hashing trick. If <= 0,
         * feature-hashing will be disabled.
         */
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
        // public boolean extraVariablesForNilAgreement = true;
        public SprlLabelConverter labelConverter = new BinarySprlLabelConverter(3.5);
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
    private SrlFeatureExtractor obsFe;
    private SprlVar[][][] sprlVars;
    private CorpusStatistics cs;

    // private Var[][] argVars = null;
    private BiasOnlyObsFeatureExtractor biasOnlyFe;

    public SprlFactorGraphBuilder(SprlFactorGraphBuilderPrm prm) {
        this.prm = prm;
    }

    public SprlVar[][][] getSprlVars() {
        return sprlVars;
    }

    public void build(IntAnnoSentence isent, ObsFeatureConjoiner ofc, FactorGraph fg, CorpusStatistics cs) {
        build(isent.getAnnoSentence(), ofc, fg, cs, new SrlFeatureExtractor(prm.srlFePrm, isent, cs, ofc));
    }

    // TODO: I want to record in the sentence pred,arg,property triples should
    // be marginalized over
    /**
     * Adds factors and variables to the given factor graph.
     */
    public void build(AnnoSentence sent, ObsFeatureConjoiner ofc, FactorGraph fg, CorpusStatistics cs,
            SrlFeatureExtractor fe) {
        // NOTE: setting this to null does not set automatically!
        VarType sprlType = VarType.PREDICTED;
        ArrayList<String> sprlStateNames = new ArrayList<>(cs.sprlStateNames);
        if (prm.roleStructure != RoleStructure.PAIRS_GIVEN) {
            sprlStateNames.add(SprlLabelConverter.nil());
        }
        this.biasOnlyFe = new BiasOnlyObsFeatureExtractor(ofc, prm.srlFePrm.featureHashMod);
        this.obsFe = fe;
        this.cs = cs;
        // should we take note of feature hash mod?

        // create the variables
        int n = sent.size();
        sprlVars = new SprlVar[n][n][cs.sprlPropertyNames.size()];

        // loop over the pred-arg pairs that will get sprl labels assigned
        for (Pair<Integer, Integer> e : pairsToLabel(sent)) {
            int i = e.get1();
            int j = e.get2();

            // make a sprl property variable for each pair-property combo
            for (Indexed<String> q1 : Indexed.enumerate(cs.sprlPropertyNames)) {
                String name = "sprl_r" + i + "-" + "a" + j + "_" + q1.get();
                SprlVar v1 = new SprlVar(sprlType, sprlStateNames.size(), name, sprlStateNames, i, j);
                fg.addVar(v1);
                sprlVars[i][j][q1.index()] = v1;

                // Add unary factors on properties.
                if (prm.unaryFactors) {
                    VarSet vars = new VarSet(v1);
                    fg.addFactor(new ObsFeTypedFactor(vars, SprlFactorType.SPRL_UNARY,
                            makeKey(SprlFactorType.SPRL_UNARY, q1.get()), ofc, obsFe));
                }
                // add pairwise properties
                if (prm.pairwiseFactors) {
                    // pick an earlier property since we want distinct pairs
                    // with both vars built
                    for (Indexed<String> q2 : Indexed.enumerate(cs.sprlPropertyNames.subList(0, q1.index()))) {
                        SprlVar v2 = sprlVars[i][j][q2.index()];
                        // factor is between v1 and an earlier one
                        fg.addFactor(new ObsFeTypedFactor(new VarSet(v1, v2), SprlFactorType.SPRL_PAIRWISE,
                                makeKey(SprlFactorType.SPRL_PAIRWISE, q1.get(), q2.get()), ofc, biasOnlyFe));
                    }
                }
            }
        }
    }

    public Iterable<Pair<Integer, Integer>> pairsToLabel(AnnoSentence s) {
        return SrlFactorGraphBuilder.getPossibleRolePairs(s, prm.roleStructure, prm.allowPredArgSelfLoops, false);
    }

    public SrlFeatureExtractor getFeatExtractor() {
        return obsFe;
    }

    /**
     * Use the variable assignment in varConfig to annotate the given sentence
     * with sprl
     */
    public void configToAnno(VarConfig varConfig, AnnoSentence toAnnotate) {
        SprlProperties sprl = new SprlProperties(prm.labelConverter);
        for (Pair<Integer, Integer> e : pairsToLabel(toAnnotate)) {
            int pred = e.get1();
            int arg = e.get2();
            for (Indexed<String> q : Indexed.enumerate((cs.sprlPropertyNames))) {
                Var sprlVar = sprlVars[pred][arg][q.index()];
                String sprlLabel = varConfig.getStateName(sprlVar);
                sprl.set(pred, arg, q.get(), sprlLabel);
            }
        }
        toAnnotate.setSprl(sprl);
        // only overwrite the pairs if they were not given
        if (prm.roleStructure != RoleStructure.PAIRS_GIVEN) {
            toAnnotate.setKnownSprlPairs(sprl.getPairs());
        }
        // only overwrite the preds if neither pairs nor preds were given
        if (prm.roleStructure != RoleStructure.PREDS_GIVEN && prm.roleStructure != RoleStructure.PAIRS_GIVEN) {
            toAnnotate.setKnownSprlPreds(sprl.getPreds());
        }
    }

    // encode
    // add sprl variables with values as found in the gold sentence
    public void annoToConfig(AnnoSentence goldSent, VarConfig addTo) {
        // we will create some sprl variables possibly looking at the known sprl
        // predicates and pairs
        SprlProperties props = goldSent.getSprl();
        for (Pair<Integer, Integer> e : pairsToLabel(goldSent)) {
            int pred = e.get1();
            int arg = e.get2();
            // TODO: handle latent sprl
            boolean isAnArg = props.containsPair(e);
            for (Indexed<String> q : Indexed.enumerate(cs.sprlPropertyNames)) {
                // add the variable to the config
                Var sprlVar = sprlVars[pred][arg][q.index()];
                addTo.put(sprlVar, isAnArg ? props.get(pred, arg, q.get()) : SprlLabelConverter.nil());
            }
        }
    }

    public static void addSprlSrlFactors(AnnoSentence sent, ObsFeatureConjoiner ofc, CorpusStatistics cs,
            FactorGraph fg, SprlFactorGraphBuilder sprl, SrlFactorGraphBuilder srl, boolean sprlPairs) {
        boolean hasSprl = sprl != null;
        boolean hasSrl = srl != null;

        // we only bother about the information between sprl and srl if one of
        // the two is present
        if (hasSprl || hasSrl) {
            SprlFactorGraphBuilderPrm sprlPrm = hasSprl ? sprl.prm : null;
            SrlFactorGraphBuilderPrm srlPrm = hasSrl ? srl.getPrm() : null;
            BiasOnlyObsFeatureExtractor biasOnlyFe = hasSprl ? sprl.biasOnlyFe
                    : new BiasOnlyObsFeatureExtractor(ofc, srlPrm.srlFePrm.featureHashMod);

            // if both are present, they must agree on the structure
            if (hasSprl && hasSrl && srlPrm.roleStructure != sprlPrm.roleStructure) {
                throw new IllegalArgumentException(
                        "cannot put factors between sprl and srl if they have differen't role structure");
            }
            if (hasSprl && hasSrl && srlPrm.allowPredArgSelfLoops != sprlPrm.allowPredArgSelfLoops) {
                throw new IllegalArgumentException(
                        "cannot put factors between sprl and srl if they don't agree on predArgSelfLoops");
            }

            // set the roleStructure
            RoleStructure rS = hasSrl ? srlPrm.roleStructure : sprlPrm.roleStructure;
            boolean allowSelfLoops = hasSrl ? srlPrm.allowPredArgSelfLoops : sprlPrm.allowPredArgSelfLoops;
            RoleVar[][] roleVars = srl != null ? srl.getRoleVars() : null;
            SprlVar[][][] sprlVars = sprl != null ? sprl.getSprlVars() : null;

            ObsFeatureExtractor fe = null;
            // TODO: add featurize back in
            // if (featurize) {
            // fe = sprl != null ? sprl.getFeatExtractor() :
            // srl.getFeatExtractor();
            // } else {
            fe = biasOnlyFe;
            // }
            // TODO: add nil agreement back in
            // boolean enforceSprlNilAgreement = hasSprl && enforceNilAgreement
            // && rS != RoleStructure.PAIRS_GIVEN;
            addSprlSrlFactorsAux(sent, ofc, fg, fe, cs.sprlPropertyNames, roleVars, sprlVars, rS, allowSelfLoops,
                    sprlPairs, biasOnlyFe);
        }

    }

    /**
     * Adds factors between srl roles and sprl property variables; if either srl
     * or sprl is not included in the model, then adding these factors ammounts
     * to adding unary factors on the included variables that reflect the gold
     * label for the corresponding variables that are not included; nilAgreement
     * is optionally enforced
     */
    private static void addSprlSrlFactorsAux(AnnoSentence sent, ObsFeatureConjoiner ofc, FactorGraph fg,
            ObsFeatureExtractor fe, List<String> propNames, RoleVar[][] roleVars, SprlVar[][][] sprlVars,
            RoleStructure rS, boolean allowSelfLoops, boolean sprlPairs, BiasOnlyObsFeatureExtractor biasOnlyFe) {
        boolean givenSrl = roleVars == null;
        boolean givenSprl = sprlVars == null;
        SprlProperties sprl = sent.getSprl();

        for (Pair<Integer, Integer> e : SrlFactorGraphBuilder.getPossibleRolePairs(sent, rS, allowSelfLoops,
                roleVars != null)) {
            int i = e.get1(), j = e.get2();

            // Get either the role variable or the gold role label
            RoleVar roleVar = null;
            String goldSrl = null;
            if (givenSrl) {
                SrlEdge srlEdge = sent.getSrlGraph().toSrlGraph().getEdge(i, j);
                goldSrl = srlEdge != null ? srlEdge.getLabel() : RoleVar.getNilStateName();
            } else {
                roleVar = roleVars[i][j];
            }

            for (Indexed<String> q1 : Indexed.enumerate(propNames)) {
                // Get either the sprl variable or the gold sprl label
                SprlVar sprlVar = null;
                String goldSprl = null;
                if (givenSprl) {
                    goldSprl = sprl.get(i, j, q1.get());
                } else {
                    sprlVar = sprlVars[i][j][q1.index()];
                }

                // add factors that look at at the current i,j,q triple
                addSrlSprlFactor(ofc, fg, fe, q1.get(), roleVar, goldSrl, sprlVar, goldSprl);

                // add a factor that looks at gold obs sprl pairs
                if (givenSprl && sprlPairs) {
                    JointFactorTemplate ft = JointFactorTemplate.ROLE_SPRL_SPRL;
                    // pick an earlier other property to conjoin with
                    for (Indexed<String> q2 : Indexed.enumerate(propNames.subList(0, q1.index()))) {
                        Serializable templateKey = makeKey(ft, "GOLD_SPRL_PAIR", q1.get(), goldSprl, q2.get(),
                                sprl.get(i, j, q2.get()));
                        // TODO: do more features help here?
                        fg.addFactor(new ObsFeTypedFactor(new VarSet(roleVar), ft, templateKey, ofc, biasOnlyFe));
                    }
                }
            }
        }
    }

    /**
     * Adds factors to account for the interaction between the sprl and srl for
     * a particular (predicate, argument, property) triple; if both variables
     * are non-null, then a pariwise factor is created between them (optionally
     * enforcing agreement on whether or not the predication of both should or
     * should not be NOT_AN_ARG) if only one is non-null, then a unary factor is
     * placed on that variable with features tied to the gold value of the null
     * variable
     */
    private static void addSrlSprlFactor(ObsFeatureConjoiner ofc, FactorGraph fg, ObsFeatureExtractor fe, String q,
            RoleVar roleV, String goldSrl, SprlVar sprlV, String goldSprl) {
        JointFactorTemplate ft = JointFactorTemplate.ROLE_SPRL_BINARY;
        if (roleV == null) { // given sprlGsrl
            fg.addFactor(new ObsFeTypedFactor(new VarSet(sprlV), ft, makeKey(ft, q, "GOLD_SRL", goldSrl), ofc, fe));
        } else if (sprlV == null) { // srlGsprl
            fg.addFactor(new ObsFeTypedFactor(new VarSet(roleV), ft, makeKey(ft, "GOLD_SPRL", q, goldSprl), ofc, fe));
        } else { // joint factor
            Serializable templateKey = makeKey(ft, q);
            fg.addFactor(new ObsFeTypedFactor(new VarSet(roleV, sprlV), ft, templateKey, ofc, fe));
        }
    }

}
