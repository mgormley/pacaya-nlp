package edu.jhu.nlp.sprl;

import static edu.jhu.nlp.joint.JointNlpFactorGraph.makeKey;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.CorpusStatistics;
import edu.jhu.nlp.Indexed;
import edu.jhu.nlp.ObsFeTypedFactor;
import edu.jhu.nlp.data.conll.SrlGraph.SrlEdge;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.data.simple.IntAnnoSentence;
import edu.jhu.nlp.joint.JointNlpFactorGraph.IsArgLabel;
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
import edu.jhu.pacaya.util.Prm;
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
    private SrlFeatureExtractor obsFe = null;
    private SprlVar[][][] sprlVars;
    private CorpusStatistics cs = null;
    
    private Var[][] argVars = null;
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
    
    //TODO: I want record in the sentence pred,arg,property triples should be marginalized over
    /**
     * Adds factors and variables to the given factor graph.
     */
    public void build(AnnoSentence sent, ObsFeatureConjoiner ofc, FactorGraph fg, CorpusStatistics cs,
            SrlFeatureExtractor fe) {
        this.biasOnlyFe = new BiasOnlyObsFeatureExtractor(ofc, prm.srlFePrm.featureHashMod);
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
        }
        ArrayList<String> sprlStateNames = new ArrayList<>(cs.sprlStateNames);
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
                SprlVar v = new SprlVar(sprlType, sprlStateNames.size(), name, sprlStateNames,
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
// TODO: add back nil agreement
//                if (prm.extraVariablesForNilAgreement) {
//                     TODO: do more feature help here?
//                    fg.addFactor(new ObsFeTypedFactorWithNilAgreement(Arrays.asList(argVars[i][j], v),
//                            Arrays.asList(IsArgLabel.NOT_AN_ARG.ordinal(), SprlClassLabel.NOT_AN_ARG.ordinal()),
//                            ISARG_SPRL_BINARY, makeKey(ISARG_SPRL_BINARY, q), ofc, biasOnlyFe));
//                }
            }
            if (prm.pairwiseFactors) {
                for (int qix1 = 0; qix1 < cs.sprlPropertyNames.size(); qix1++) {
                    String q1 = cs.sprlPropertyNames.get(qix1);
                    SprlVar v1 = sprlVars[i][j][qix1];
                    for (int qix2 = 0; qix2 < cs.sprlPropertyNames.size(); qix2++) {
                        String q2 = cs.sprlPropertyNames.get(qix2);
                        SprlVar v2 = sprlVars[i][j][qix2];
                        // TOOD: do more feature help here?
                        fg.addFactor(new ObsFeTypedFactor(new VarSet(v1, v2),
                                SprlFactorType.SPRL_PAIRWISE,
                                makeKey(SprlFactorType.SPRL_PAIRWISE, q1, q2),
                                ofc, biasOnlyFe));
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
        SprlProperties sprl = new SprlProperties(prm.labelConverter);
        for (Pair<Integer, Integer> e : SrlFactorGraphBuilder.getPossibleRolePairs(toAnnotate.size(),
                toAnnotate.getKnownSprlPreds(), toAnnotate.getKnownSprlPairs(), toAnnotate.getPairsToSkip(),
                prm.roleStructure, prm.allowPredArgSelfLoops)) {
            int pred = e.get1();
            int arg = e.get2();
            // we don't need to worry about decoding the argVar since we get
            // that information from the sprl anyway
            for (Indexed<String> q : Indexed.enumerate((cs.sprlPropertyNames))) {
                Var sprlVar = sprlVars[pred][arg][q.index()];
                String sprlLabel = varConfig.getStateName(sprlVar);
                sprl.set(pred, arg, q.get(), sprlLabel);
            }
        }
        toAnnotate.setSprl(sprl);
        // only overwrite the pairs if they were not given
        if (prm.roleStructure != RoleStructure.PAIRS_GIVEN) {
            toAnnotate.setKnownSprlPairs(sprl.getKnownPairs());
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
        for (Pair<Integer, Integer> e : SrlFactorGraphBuilder.getPossibleRolePairs(goldSent.size(),
                goldSent.getKnownSprlPreds(), goldSent.getKnownSprlPairs(), goldSent.getPairsToSkip(),
                prm.roleStructure, prm.allowPredArgSelfLoops)) {
            int pred = e.get1();
            int arg = e.get2();
            // TODO: handle latent sprl
            boolean isAnArg = props.containsPair(e);

            if (prm.extraVariablesForNilAgreement) {
                addTo.put(argVars[pred][arg], isAnArg ? IsArgLabel.IS_ARG.ordinal() : IsArgLabel.NOT_AN_ARG.ordinal());
            }
            for (Indexed<String> q : Indexed.enumerate(cs.sprlPropertyNames)) {
                // add the variable to the config
                Var sprlVar = sprlVars[pred][arg][q.index()];
                addTo.put(sprlVar, isAnArg ? props.get(pred, arg, q.get()) : SprlLabelConverter.nil());
            }
        }
    }

    public static void addSprlSrlFactors(AnnoSentence sent, ObsFeatureConjoiner ofc, CorpusStatistics cs, FactorGraph fg,
            SprlFactorGraphBuilder sprl, SrlFactorGraphBuilder srl,
            boolean sprlPairs, boolean enforceNilAgreement, boolean featurize) {
        boolean hasSprl = sprl != null;
        boolean hasSrl = srl != null;
        
        // we only bother about the information between sprl and srl if one of the two is present
        if (hasSprl || hasSrl) {
            SprlFactorGraphBuilderPrm sprlPrm = hasSprl ? sprl.prm : null;
            SrlFactorGraphBuilderPrm srlPrm = hasSrl ? srl.getPrm() : null;
            BiasOnlyObsFeatureExtractor biasOnlyFe = hasSprl ? sprl.biasOnlyFe : new BiasOnlyObsFeatureExtractor(ofc, srlPrm.srlFePrm.featureHashMod);
            
            // if both are present, they must agree on the structure
            if (hasSprl && hasSrl && srlPrm.roleStructure != sprlPrm.roleStructure) {
                throw new IllegalArgumentException("cannot put factors between sprl and srl if they have differen't role structure");
            }
            if (hasSprl && hasSrl && srlPrm.allowPredArgSelfLoops != sprlPrm.allowPredArgSelfLoops) {
                throw new IllegalArgumentException("cannot put factors between sprl and srl if they don't agree on predArgSelfLoops");
            }

            // set the roleStructure
            RoleStructure rS = hasSrl ? srlPrm.roleStructure : sprlPrm.roleStructure;
            boolean allowSelfLoops = hasSrl ? srlPrm.allowPredArgSelfLoops : sprlPrm.allowPredArgSelfLoops;
            RoleVar[][] roleVars = srl != null ? srl.getRoleVars() : null;
            SprlVar[][][] sprlVars = sprl != null ? sprl.getSprlVars() : null;

            ObsFeatureExtractor fe = null;
            if (featurize) {
                fe = sprl != null ? sprl.getFeatExtractor() : srl.getFeatExtractor();
            } else {
                fe = biasOnlyFe;
            }
            boolean enforceSprlNilAgreement = hasSprl && enforceNilAgreement && rS != RoleStructure.PAIRS_GIVEN;
            addSrlSprlFactors(sent, ofc, fg, fe, cs.sprlPropertyNames, roleVars, sprlVars, rS, allowSelfLoops,
                    sprlPairs, enforceSprlNilAgreement, biasOnlyFe);
        }
        
    }

    /**
     * Adds factors between srl roles and sprl property variables;
     * if either srl or sprl is not included in the model, then adding these factors
     * ammounts to adding unary factors on the included variables that reflect the gold label
     * for the corresponding variables that are not included; nilAgreement is optionally enforced 
     */
    private static void addSrlSprlFactors(AnnoSentence sent, ObsFeatureConjoiner ofc, FactorGraph fg,
            ObsFeatureExtractor fe, List<String> propNames, RoleVar[][] roleVars, SprlVar[][][] sprlVars,
            RoleStructure rS, boolean allowSelfLoops, boolean sprlPairs, boolean enforceNilAgreement, BiasOnlyObsFeatureExtractor biasOnlyFe) {
        boolean givenSrl = roleVars == null ;
        boolean givenSprl = sprlVars == null ;
        assert !givenSrl || !givenSprl;
        SprlProperties sprl = sent.getSprl();
        
        for (Pair<Integer, Integer> e : SrlFactorGraphBuilder.getPossibleRolePairs(sent, rS, allowSelfLoops, roleVars != null)) {
            int i = e.get1(), j = e.get2(); 

            // Get either the role variable or the gold role label
            RoleVar roleVar = null;
            String goldSrl = null;
            if (givenSrl) {
                SrlEdge srlEdge = sent.getSrlGraph().getEdge(i, j); 
                goldSrl = srlEdge != null ? srlEdge.getLabel() : RoleVar.getNilStateName();
            } else {
                roleVar = roleVars[i][j];
            }

            for (Indexed<String> q : Indexed.enumerate(propNames)) {
                // Get either the sprl variable or the gold sprl label
                SprlVar sprlVar = null;
                String goldSprl = null;
                if (givenSprl) {
                    goldSprl = sprl.get(i, j, q.get());
                } else {
                    sprlVar = sprlVars[i][j][q.index()];
                }

                // add factors that look at at the current i,j,q triple
                addSrlSprlFactor(ofc, fg, fe, q.get(), roleVar, goldSrl, sprlVar, goldSprl, enforceNilAgreement);

                // for observed sprlPairs, add a factor that looks at gold sprl pairs 
                if (givenSprl && sprlPairs) {
                    for (Indexed<String> q2 : Indexed.enumerate(propNames)) {
                        JointFactorTemplate ft = JointFactorTemplate.ROLE_SPRL_SPRL;
                        Serializable templateKey = makeKey(ft, "GOLD_SPRL_PAIR", q.get(), goldSprl, q2.get(), sprl.get(i, j, q2.get()));
                        // TODO: do more features help here?
                        fg.addFactor(new ObsFeTypedFactor(new VarSet(roleVar), ft, templateKey, ofc, biasOnlyFe));
                    }
                }
            }
        }
    }

    /**
     * Adds factors to account for the interaction between the sprl and srl for a particular
     * (predicate, argument, property) triple; if both variables are non-null, then a pariwise factor is created between them
     * (optionally enforcing agreement on whether or not the predication of both should or should not
     * be NOT_AN_ARG)
     * if only one is non-null, then a unary factor is placed on that variable with features tied to the
     * gold value of the null variable 
     */
    private static void addSrlSprlFactor(ObsFeatureConjoiner ofc, FactorGraph fg, ObsFeatureExtractor fe,
            String q, RoleVar roleV, String goldSrl, SprlVar sprlV, String goldSprl,
            boolean enforceNilAgreement) {
        JointFactorTemplate ft = JointFactorTemplate.ROLE_SPRL_BINARY;
        if (roleV == null) { // given sprlGsrl
            fg.addFactor(new ObsFeTypedFactor(new VarSet(sprlV), ft, makeKey(ft, q, "GOLD_SRL", goldSrl), ofc, fe));
        } else if (sprlV == null) { // srlGsprl
            fg.addFactor(new ObsFeTypedFactor(new VarSet(roleV), ft, makeKey(ft, q, "GOLD_SPRL", goldSprl), ofc, fe));
        } else { // joint factor
            Serializable templateKey = makeKey(ft, q); 
            // real pairwise factors
// TODO: add back nil agreement
//            if (enforceNilAgreement) { // create the factor in such a way that nil agreement is enforced
//                fg.addFactor(new ObsFeTypedFactorWithNilAgreement(Arrays.asList(roleV, sprlV),
//                        Arrays.asList(roleV.getNilState(), SprlClassLabel.NOT_AN_ARG.ordinal()),
//                        ft, templateKey, ofc, fe));
//            } else { // ordinary pairwise factor that doesn't enforce nil agreement
            fg.addFactor(new ObsFeTypedFactor(new VarSet(roleV, sprlV), ft, templateKey, ofc, fe));
//            }
        }
    }

//    private static SprlClassLabel goldSprlLabel(AnnoSentence sent, Pair<Integer, Integer> e, String q) {
//        Properties props = sent.getSprl().get(e);
//        return props == null ? SprlClassLabel.NOT_AN_ARG : props.getLabel(q);
//    }



}
