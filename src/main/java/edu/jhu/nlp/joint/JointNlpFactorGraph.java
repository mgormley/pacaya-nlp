package edu.jhu.nlp.joint;

import static edu.jhu.pacaya.sch.util.Indexed.enumerate;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.CorpusStatistics;
import edu.jhu.nlp.ObsFeTypedFactor;
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
import edu.jhu.nlp.sprl.SprlFactorGraphBuilder;
import edu.jhu.nlp.sprl.SprlFactorGraphBuilder.SprlFactorGraphBuilderPrm;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.RoleVar;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.SenseVar;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.SrlFactorGraphBuilderPrm;
import edu.jhu.nlp.tag.PosTagFactorGraphBuilder;
import edu.jhu.nlp.tag.PosTagFactorGraphBuilder.PosTagFactorGraphBuilderPrm;
import edu.jhu.nlp.tag.TemplateFeatureFactor;
import edu.jhu.pacaya.gm.feat.ObsFeatureConjoiner;
import edu.jhu.pacaya.gm.model.FactorGraph;
import edu.jhu.pacaya.gm.model.Var;
import edu.jhu.pacaya.gm.model.VarSet;
import edu.jhu.pacaya.gm.model.globalfac.LinkVar;
import edu.jhu.pacaya.sch.util.Indexed;
import edu.jhu.pacaya.util.Prm;
import edu.jhu.pacaya.util.collections.QLists;

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
        // this enforcement may happen by way of specializing the srlSprl factors or by adding
        // additional variables if srl isn't being modeled
        public boolean enforceSprlNilAgreement = true;
        public boolean featurizeSrlSprlPairwise = false;
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

        // SPRL variables and factors that only touch SPRL variables
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
        if (prm.includeDp && prm.includeSrl) {
            addDpSrlFactors(ofc, fg);
        }
        if (prm.includePos && prm.includeSrl) {
            addPosSrlFactors(sent, ofc, fg, cs);
        }
        // SPRL interaction with SRL (including observed SPRL or observed SRL on SPRL)
        if (prm.sprlSrlFactors) {
            SprlFactorGraphBuilder.addSprlSrlFactors(sent, ofc, cs, fg, sprl, srl, prm.sprlPrm.pairwiseFactors, prm.enforceSprlNilAgreement, prm.featurizeSrlSprlPairwise);
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
