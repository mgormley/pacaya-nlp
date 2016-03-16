package edu.jhu.nlp.joint;

import static edu.jhu.nlp.Indexed.enumerate;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.CorpusStatistics;
import edu.jhu.nlp.Indexed;
import edu.jhu.nlp.ObsFeTypedFactor;
import edu.jhu.nlp.ObsFeTypedFactorWithNilAgreement;
import edu.jhu.nlp.data.Properties;
import edu.jhu.nlp.data.conll.SrlGraph.SrlEdge;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.data.simple.IntAnnoSentence;
import edu.jhu.nlp.depparse.DepParseFactorGraphBuilder;
import edu.jhu.nlp.depparse.DepParseFactorGraphBuilder.DepParseFactorGraphBuilderPrm;
import edu.jhu.nlp.features.LocalObservations;
import edu.jhu.nlp.features.TemplateFeatureExtractor;
import edu.jhu.nlp.features.TemplateLanguage.FeatTemplate;
import edu.jhu.nlp.features.TemplateLanguage.FeatTemplate1;
import edu.jhu.nlp.features.TemplateLanguage.Position;
import edu.jhu.nlp.features.TemplateLanguage.PositionModifier;
import edu.jhu.nlp.features.TemplateLanguage.TokProperty;
import edu.jhu.nlp.relations.RelationsFactorGraphBuilder;
import edu.jhu.nlp.relations.RelationsFactorGraphBuilder.RelationsFactorGraphBuilderPrm;
import edu.jhu.nlp.sprl.SprlClassLabel;
import edu.jhu.nlp.sprl.SprlFactorGraphBuilder;
import edu.jhu.nlp.sprl.SprlFactorGraphBuilder.SprlFactorGraphBuilderPrm;
import edu.jhu.nlp.sprl.SprlFactorGraphBuilder.SprlVar;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.RoleStructure;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.RoleVar;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.SenseVar;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.SrlFactorGraphBuilderPrm;
import edu.jhu.nlp.tag.PosTagFactorGraphBuilder;
import edu.jhu.nlp.tag.PosTagFactorGraphBuilder.PosTagFactorGraphBuilderPrm;
import edu.jhu.nlp.tag.TemplateFeatureFactor;
import edu.jhu.pacaya.gm.feat.ObsFeatureConjoiner;
import edu.jhu.pacaya.gm.feat.ObsFeatureExtractor;
import edu.jhu.pacaya.gm.model.FactorGraph;
import edu.jhu.pacaya.gm.model.Var;
import edu.jhu.pacaya.gm.model.VarSet;
import edu.jhu.pacaya.gm.model.globalfac.LinkVar;
import edu.jhu.pacaya.util.Prm;
import edu.jhu.pacaya.util.collections.QLists;
import edu.jhu.prim.tuple.Pair;

/**
 * A factor graph builder for joint dependency parsing and semantic role
 * labeling. Note this class also extends FactorGraph in order to provide easy
 * lookups of cached variables.
 *
 * @author mmitchell
 * @author mgormley
 */
public class JointNlpFactorGraph extends FactorGraph {

    public enum IsArgLabel {
        IS_ARG, NOT_AN_ARG;
        // add the labels as string names
        public static ArrayList<String> labels;

        static {
            labels = new ArrayList<>();
            for (IsArgLabel label : values()) {
                labels.add(label.name());
            }
        }
    }

    private static final long serialVersionUID = 1L;

    private static final Logger log = LoggerFactory.getLogger(JointNlpFactorGraph.class);

    /**
     * Parameters for the {@link JointNlpFactorGraph}.
     *
     * @author mgormley
     */
    public static class JointNlpFactorGraphPrm extends Prm {
        private static final long serialVersionUID = 1L;
        public boolean includePos = false;
        public PosTagFactorGraphBuilderPrm posPrm = new PosTagFactorGraphBuilderPrm();
        public boolean includeDp = true;
        public DepParseFactorGraphBuilderPrm dpPrm = new DepParseFactorGraphBuilderPrm();
        public boolean includeSrl = true;
        public SrlFactorGraphBuilderPrm srlPrm = new SrlFactorGraphBuilderPrm();
        public boolean includeRel = false;
        public RelationsFactorGraphBuilderPrm relPrm = new RelationsFactorGraphBuilderPrm();
        /** Whether to use SRL feats for Link-Role factors. */
        public boolean useSrlFeatsForLinkRoleFactors = true;
        /** Whether to include SPRL */
        public boolean includeSprl = false;
        public SprlFactorGraphBuilderPrm sprlPrm = new SprlFactorGraphBuilderPrm();
        public boolean sprlSrlFactors = false;
        public boolean enforceSprlNilAgreement = true;
    }

    public static LinkedList<Serializable> makeKey(Serializable... args) {
        return new LinkedList<Serializable>(Arrays.asList(args));
    }


    public enum JointFactorTemplate {
        LINK_ROLE_BINARY, ROLE_P_TAG_BINARY, ROLE_C_TAG_BINARY, ROLE_SPRL_BINARY, ISARG_SPRL_BINARY, ROLE_SPRL_SPRL
    }

    // Parameters for constructing the factor graph.
    private JointNlpFactorGraphPrm prm;

    // The sentence length.
    private int n;

    // Factor graph builders, which also cache the variables.
    private PosTagFactorGraphBuilder pos;
    private DepParseFactorGraphBuilder dp;
    private SrlFactorGraphBuilder srl;
    private RelationsFactorGraphBuilder rel;
    private SprlFactorGraphBuilder sprl;

    public JointNlpFactorGraph(JointNlpFactorGraphPrm prm, AnnoSentence sent, CorpusStatistics cs,
            ObsFeatureConjoiner ofc) {
        this.prm = prm;
        build(sent, cs, ofc, this);
    }

    /**
     * Adds factors and variables to the given factor graph.
     */
    public void build(AnnoSentence sent, CorpusStatistics cs, ObsFeatureConjoiner ofc, FactorGraph fg) {
        this.n = sent.size();

        // TODO: This should move up the stack.
        IntAnnoSentence isent = new IntAnnoSentence(sent, cs.store);

        // if we have sprl variables but they won't be connected to srl variables, then we need
        // to make variables to enforce nil agreement
        //boolean includeIsArg = prm.includeSprl && prm.sprlPrm.enforceSprlNilAgreement && !(prm.sprlSrlFactors && prm.includeSrl); 

        if (prm.includeSprl) {
            sprl = new SprlFactorGraphBuilder(prm.sprlPrm);
            sprl.build(isent, ofc, fg, cs);
        }
        if (prm.includePos) {
            pos = new PosTagFactorGraphBuilder(prm.posPrm);
            pos.build(isent, ofc, fg, cs);
        }
        if (prm.includeDp) {
            dp = new DepParseFactorGraphBuilder(prm.dpPrm);
            dp.build(isent, fg, cs, ofc);
        }
        if (prm.includeSrl) {
            srl = new SrlFactorGraphBuilder(prm.srlPrm);
            srl.build(isent, cs, ofc, fg);
        }
        if (prm.includeRel) {
            rel = new RelationsFactorGraphBuilder(prm.relPrm);
            rel.build(sent, ofc, fg, cs);
        }
        if (prm.sprlSrlFactors && (prm.includeSprl || prm.includeSrl)) {
            // we only bother about the information between sprl and srl if one of the two is present

            // set the roleStructure and make sure it is consistent
            RoleStructure rS = prm.includeSrl ? prm.srlPrm.roleStructure : prm.sprlPrm.roleStructure;
            boolean allowSelfLoops = prm.includeSrl ? prm.srlPrm.allowPredArgSelfLoops : prm.sprlPrm.allowPredArgSelfLoops;
            if (prm.includeSrl && prm.includeSprl) {
                assert prm.srlPrm.roleStructure == prm.sprlPrm.roleStructure;
                assert prm.srlPrm.allowPredArgSelfLoops == prm.sprlPrm.allowPredArgSelfLoops;
            }

            RoleVar[][] roleVars = prm.includeSrl ? srl.getRoleVars() : null;
            SprlVar[][][] sprlVars = prm.includeSprl ? sprl.getSprlVars() : null;
            ObsFeatureExtractor fe = prm.includeSprl ? sprl.getFeatExtractor() : srl.getFeatExtractor();
            boolean enforceNilAgreement = prm.enforceSprlNilAgreement && (rS != RoleStructure.PAIRS_GIVEN);
            
            addSrlSprlFactors(sent, ofc, fg, fe, cs.sprlPropertyNames, roleVars, sprlVars, rS, allowSelfLoops,
                    prm.sprlPrm.pairwiseFactors, enforceNilAgreement);
        }
        if (prm.includeDp && prm.includeSrl) {
            addDpSrlFactors(ofc, fg);
        }
        if (prm.includePos && prm.includeSrl) {
            addPosSrlFactors(sent, ofc, fg, cs);
        }
    }

    private void addPosSrlFactors(AnnoSentence sent, ObsFeatureConjoiner ofc, FactorGraph fg, CorpusStatistics cs) {
        // Add the joint factors.
        TemplateFeatureExtractor fe = new TemplateFeatureExtractor(sent, cs);
        List<FeatTemplate> templates = QLists.getList(
                new FeatTemplate1(Position.PARENT, PositionModifier.IDENTITY, TokProperty.WORD), // word(p)
                new FeatTemplate1(Position.CHILD, PositionModifier.IDENTITY, TokProperty.WORD), // word(c)
                new FeatTemplate1(Position.PARENT, PositionModifier.IDENTITY, TokProperty.BC0), // bc0(p)
                new FeatTemplate1(Position.CHILD, PositionModifier.IDENTITY, TokProperty.BC0) // bc0(c)
        );
        List<Var> tagVars = pos.getTagVars();
        RoleVar[][] roleVars = srl.getRoleVars();
        for (int i = -1; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i != -1) {
                    // Add binary factors between Roles and Tags.
                    if (roleVars[i][j] != null) {
                        LocalObservations local = LocalObservations.newPidxCidx(i, j);
                        if (tagVars.get(i) != null) {
                            fg.addFactor(new TemplateFeatureFactor(new VarSet(roleVars[i][j], tagVars.get(i)),
                                    JointFactorTemplate.ROLE_P_TAG_BINARY, ofc, local, fe, templates,
                                    prm.srlPrm.srlFePrm.featureHashMod));
                        }
                        if (tagVars.get(j) != null) {
                            fg.addFactor(new TemplateFeatureFactor(new VarSet(roleVars[i][j], tagVars.get(j)),
                                    JointFactorTemplate.ROLE_C_TAG_BINARY, ofc, local, fe, templates,
                                    prm.srlPrm.srlFePrm.featureHashMod));
                        }
                    }
                }
            }
        }

    }

    private void addDpSrlFactors(ObsFeatureConjoiner ofc, FactorGraph fg) {
        // Add the joint factors.
        LinkVar[][] childVars = dp.getChildVars();
        RoleVar[][] roleVars = srl.getRoleVars();
        for (int i = -1; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i != -1) {
                    // Add binary factors between Roles and Links.
                    if (roleVars[i][j] != null && childVars[i][j] != null) {
                        fg.addFactor(new ObsFeTypedFactor(new VarSet(roleVars[i][j], childVars[i][j]),
                                JointFactorTemplate.LINK_ROLE_BINARY, ofc, srl.getFeatExtractor()));
                    }
                }
            }
        }
    }
    
    private static SprlClassLabel goldSprlLabel(AnnoSentence sent, Pair<Integer, Integer> e, String q) {
        Properties props = sent.getSprl().get(e);
        return props == null ? SprlClassLabel.NOT_AN_ARG : props.getLabel(q);
    }

    /**
     * Adds factors between srl roles and sprl property variables;
     * if either srl or sprl is not included in the model, then adding these factors
     * ammounts to adding unary factors on the included variables that reflect the gold label
     * for the corresponding variables that are not included; nilAgreement is optionally enforced 
     */
    private static void addSrlSprlFactors(AnnoSentence sent, ObsFeatureConjoiner ofc, FactorGraph fg,
            ObsFeatureExtractor fe, List<String> propNames, RoleVar[][] roleVars, SprlVar[][][] sprlVars,
            RoleStructure rS, boolean allowSelfLoops, boolean sprlPairs, boolean enforceNilAgreement) {
        boolean givenSrl = roleVars == null ;
        boolean givenSprl = sprlVars == null ;
        assert !givenSrl || !givenSprl;
        
        for (Pair<Integer, Integer> e : SrlFactorGraphBuilder.getPossibleRolePairs(sent, rS, allowSelfLoops, roleVars != null)) {
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

            for (Indexed<String> q : enumerate(propNames)) {
                // Get either the sprl variable or the gold sprl label
                SprlVar sprlVar = null;
                SprlClassLabel goldSprl = null;
                if (givenSprl) {
                    goldSprl = goldSprlLabel(sent, e, q.get());
                } else {
                    sprlVar = sprlVars[i][j][q.index()];
                }

                // add factors that look at at the current i,j,q triple
                addSrlSprlFactor(ofc, fg, fe, q.get(), roleVar, goldSrl, sprlVar, goldSprl, enforceNilAgreement);

                // for observed sprlPairs, add a factor that looks at gold sprl pairs 
                if (givenSprl && sprlPairs) {
                    for (Indexed<String> q2 : enumerate(propNames)) {
                        SprlClassLabel goldSprl2 = goldSprlLabel(sent, e, q2.get());
                        JointFactorTemplate ft = JointFactorTemplate.ROLE_SPRL_SPRL;
                        Serializable templateKey = makeKey(ft, q.get(), goldSprl, q2.get(), goldSprl2);
                        fg.addFactor(new ObsFeTypedFactor(new VarSet(roleVar), ft, templateKey, ofc, fe));
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
            String q, RoleVar roleV, String goldSrl, SprlVar sprlV, SprlClassLabel goldSprl,
            boolean enforceNilAgreement) {
        JointFactorTemplate ft = JointFactorTemplate.ROLE_SPRL_BINARY;
        if (roleV == null) { // given sprlGsrl
            fg.addFactor(new ObsFeTypedFactor(new VarSet(sprlV), ft, makeKey(ft, q, RoleVar.class, goldSrl), ofc, fe));
        } else if (sprlV == null) { // srlGsprl
            fg.addFactor(new ObsFeTypedFactor(new VarSet(roleV), ft, makeKey(ft, q, SprlVar.class, goldSprl), ofc, fe));
        } else { // joint factor
            Serializable templateKey = makeKey(ft, q); 
            // real pairwise factors
            if (enforceNilAgreement) { // create the factor in such a way that nil agreement is enforced
                fg.addFactor(new ObsFeTypedFactorWithNilAgreement(Arrays.asList(roleV, sprlV),
                        Arrays.asList(roleV.getNilState(), SprlClassLabel.NOT_AN_ARG.ordinal()),
                        ft, templateKey, ofc, fe));
            } else { // ordinary pairwise factor that doesn't enforce nil agreement
                fg.addFactor(new ObsFeTypedFactor(new VarSet(roleV, sprlV), ft, templateKey, ofc, fe));
            }
        }
    }

    // ----------------- Creating Variables -----------------

    // ----------------- Public Getters -----------------

    /**
     * Get the link var corresponding to the specified parent and child
     * position.
     *
     * @param parent
     *            The parent word position, or -1 to indicate the wall node.
     * @param child
     *            The child word position.
     * @return The link variable or null if it doesn't exist.
     */
    public LinkVar getLinkVar(int parent, int child) {
        if (dp == null) {
            return null;
        }
        return dp.getLinkVar(parent, child);
    }

    /**
     * Gets a Role variable.
     *
     * @param i
     *            The parent position.
     * @param j
     *            The child position.
     * @return The role variable or null if it doesn't exist.
     */
    public RoleVar getRoleVar(int i, int j) {
        if (srl == null) {
            return null;
        }
        return srl.getRoleVar(i, j);
    }

    /**
     * Gets a predicate Sense variable.
     *
     * @param i
     *            The position of the predicate.
     * @return The sense variable or null if it doesn't exist.
     */
    public SenseVar getSenseVar(int i) {
        if (srl == null) {
            return null;
        }
        return srl.getSenseVar(i);
    }

    public int getSentenceLength() {
        return n;
    }

    public SprlFactorGraphBuilder getSprlBuilder() {
        return sprl;
    }

    public PosTagFactorGraphBuilder getPosTagBuilder() {
        return pos;
    }

    public DepParseFactorGraphBuilder getDpBuilder() {
        return dp;
    }

    public SrlFactorGraphBuilder getSrlBuilder() {
        return srl;
    }

    public RelationsFactorGraphBuilder getRelBuilder() {
        return rel;
    }

}
