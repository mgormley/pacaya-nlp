package edu.jhu.nlp.joint;

import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.CorpusStatistics;
import edu.jhu.nlp.ObsFeTypedFactor;
import edu.jhu.nlp.data.simple.AnnoSentence;
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
import edu.jhu.nlp.srl.SrlFactorGraphBuilder;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.RoleVar;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.SenseVar;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.SrlFactorGraphBuilderPrm;
import edu.jhu.nlp.tag.PosTagFactorGraphBuilder;
import edu.jhu.nlp.tag.PosTagFactorGraphBuilder.PosTagFactorGraphBuilderPrm;
import edu.jhu.nlp.tag.PosTagFactorGraphBuilder.TagVar;
import edu.jhu.nlp.tag.TemplateFeatureFactor;
import edu.jhu.pacaya.gm.feat.ObsFeatureConjoiner;
import edu.jhu.pacaya.gm.model.FactorGraph;
import edu.jhu.pacaya.gm.model.VarSet;
import edu.jhu.pacaya.gm.model.globalfac.LinkVar;
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

    private static final long serialVersionUID = 1L;

    private static final Logger log = LoggerFactory.getLogger(JointNlpFactorGraph.class); 

    /**
     * Parameters for the {@link JointNlpFactorGraph}.
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
    }
    
    public enum JointFactorTemplate {
        LINK_ROLE_BINARY, ROLE_P_TAG_BINARY, ROLE_C_TAG_BINARY,
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

    public JointNlpFactorGraph(JointNlpFactorGraphPrm prm, AnnoSentence sent, CorpusStatistics cs, ObsFeatureConjoiner ofc) {
        this.prm = prm;
        build(sent, cs, ofc, this);
    }

    /**
     * Adds factors and variables to the given factor graph.
     */
    public void build(AnnoSentence sent, CorpusStatistics cs, ObsFeatureConjoiner ofc,
            FactorGraph fg) {
        this.n = sent.size();

        if (prm.includePos) {
            pos = new PosTagFactorGraphBuilder(prm.posPrm);
            pos.build(sent, ofc, fg, cs);
        }
        if (prm.includeDp) {
            dp = new DepParseFactorGraphBuilder(prm.dpPrm);
            dp.build(sent, fg, cs, ofc);
        }
        if (prm.includeSrl) {
            srl = new SrlFactorGraphBuilder(prm.srlPrm); 
            srl.build(sent, cs, ofc, fg);
        }
        if (prm.includeRel ) {
            rel = new RelationsFactorGraphBuilder(prm.relPrm);
            rel.build(sent, ofc, fg, cs);
        }
        
        if (prm.includeDp && prm.includeSrl) {
            // Add the joint factors.
            LinkVar[][] childVars = dp.getChildVars();
            RoleVar[][] roleVars = srl.getRoleVars();
            for (int i = -1; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (i != -1) {
                        // Add binary factors between Roles and Links.
                        if (roleVars[i][j] != null && childVars[i][j] != null) {
                            addFactor(new ObsFeTypedFactor(new VarSet(roleVars[i][j], childVars[i][j]), 
                                    JointFactorTemplate.LINK_ROLE_BINARY, ofc, srl.getFeatExtractor()));
                        }
                    }
                }
            }
        }
        if (prm.includePos && prm.includeSrl) {
            // Add the joint factors.
            TemplateFeatureExtractor fe = new TemplateFeatureExtractor(sent, cs);
            List<FeatTemplate> templates = QLists.getList(
                    new FeatTemplate1(Position.PARENT, PositionModifier.IDENTITY, TokProperty.WORD), // word(p)
                    new FeatTemplate1(Position.CHILD, PositionModifier.IDENTITY, TokProperty.WORD), // word(c)
                    new FeatTemplate1(Position.PARENT, PositionModifier.IDENTITY, TokProperty.BC0), // bc0(p)
                    new FeatTemplate1(Position.CHILD, PositionModifier.IDENTITY, TokProperty.BC0) // bc0(c)
                    );
            List<TagVar> tagVars = pos.getTagVars();
            RoleVar[][] roleVars = srl.getRoleVars();
            for (int i = -1; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (i != -1) {
                        // Add binary factors between Roles and Tags.
                        if (roleVars[i][j] != null) {
                            LocalObservations local = LocalObservations.newPidxCidx(i, j);
                            if (tagVars.get(i) != null) {
                                addFactor(new TemplateFeatureFactor(new VarSet(roleVars[i][j], tagVars.get(i)), 
                                        JointFactorTemplate.ROLE_P_TAG_BINARY, ofc, local , fe, 
                                        templates, prm.srlPrm.srlFePrm.featureHashMod));
                            }
                            if (tagVars.get(j) != null) {
                                addFactor(new TemplateFeatureFactor(new VarSet(roleVars[i][j], tagVars.get(j)), 
                                        JointFactorTemplate.ROLE_C_TAG_BINARY, ofc, local , fe, 
                                        templates, prm.srlPrm.srlFePrm.featureHashMod));
                            }
                        }
                    }
                }
            }
        }
    }

    // ----------------- Creating Variables -----------------

    // ----------------- Public Getters -----------------
    
    /**
     * Get the link var corresponding to the specified parent and child position.
     * 
     * @param parent The parent word position, or -1 to indicate the wall node.
     * @param child The child word position.
     * @return The link variable or null if it doesn't exist.
     */
    public LinkVar getLinkVar(int parent, int child) {
        if (dp == null) { return null; }
        return dp.getLinkVar(parent, child);
    }

    /**
     * Gets a Role variable.
     * @param i The parent position.
     * @param j The child position.
     * @return The role variable or null if it doesn't exist.
     */
    public RoleVar getRoleVar(int i, int j) {
        if (srl == null) { return null; }
        return srl.getRoleVar(i, j);
    }
    
    /**
     * Gets a predicate Sense variable.
     * @param i The position of the predicate.
     * @return The sense variable or null if it doesn't exist.
     */
    public SenseVar getSenseVar(int i) {
        if (srl == null) { return null; }
        return srl.getSenseVar(i);
    }

    public int getSentenceLength() {
        return n;
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
