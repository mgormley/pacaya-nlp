package edu.jhu.nlp.tag;

import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.CorpusStatistics;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.features.LocalObservations;
import edu.jhu.nlp.features.TemplateFeatureExtractor;
import edu.jhu.nlp.features.TemplateLanguage.FeatTemplate;
import edu.jhu.nlp.features.TemplateSets;
import edu.jhu.nlp.relations.FeatureUtils;
import edu.jhu.pacaya.gm.feat.FactorTemplateList;
import edu.jhu.pacaya.gm.feat.FeatureVector;
import edu.jhu.pacaya.gm.feat.ObsCjExpFamFactor;
import edu.jhu.pacaya.gm.feat.ObsFeatureConjoiner;
import edu.jhu.pacaya.gm.model.FactorGraph;
import edu.jhu.pacaya.gm.model.Var;
import edu.jhu.pacaya.gm.model.Var.VarType;
import edu.jhu.pacaya.gm.model.VarSet;
import edu.jhu.pacaya.util.FeatureNames;
import edu.jhu.pacaya.util.Prm;

public class PosTagFactorGraphBuilder {

    private static final Logger log = LoggerFactory.getLogger(PosTagFactorGraphBuilder.class);

    public static class PosTagFactorGraphBuilderPrm extends Prm {
        // TODO: Cleanup these names: drop "use" and add "rel" prefix.
        private static final long serialVersionUID = 1L;
        /** The type of the link variables. */
        public VarType posTagVarType = VarType.LATENT;
        /** Feature templates. */
        public List<FeatTemplate> templates = TemplateSets.getFromResource(TemplateSets.custom2TagFeatsResource);
        /** The value of the mod for use in the feature hashing trick. If <= 0, feature-hashing will be disabled. */
        public int featureHashMod = -1;
    }
    
    public enum PosTagFactorType {
        POS_TAG, INIT_TAG
    }
    
    private PosTagFactorGraphBuilderPrm prm;
    private List<TagVar> tagVars;
    
    public PosTagFactorGraphBuilder(PosTagFactorGraphBuilderPrm prm) {
        this.prm = prm;
    }
    
    public static class TagVar extends Var {
        private static final long serialVersionUID = 1L;
        public int i;
        public TagVar(VarType type, String name, List<String> stateNames, int i) {
            super(type, stateNames.size(), name, stateNames);
            this.i = i;
        }
        public static String getDefaultName(int i) {
            return String.format("TagVar[%d]", i);
        }
    }
    
    public class PosTagFactor extends ObsCjExpFamFactor {

        private static final long serialVersionUID = 1L;
        private LocalObservations local;
        private TemplateFeatureExtractor fe;
        
        public PosTagFactor(VarSet vars, Object templateKey, ObsFeatureConjoiner ofc, 
                LocalObservations local, TemplateFeatureExtractor fe) {
            super(vars, templateKey, ofc);
            this.local = local;
            this.fe = fe;
        }
        
        @Override
        public FeatureVector getObsFeatures() {
            FactorTemplateList fts = ofc.getTemplates();
            final FeatureNames alphabet = fts.getTemplate(this).getAlphabet();
            ArrayList<String> obsFeats = new ArrayList<String>();
            fe.addFeatures(prm.templates, local, obsFeats);
            
            // The bias features are used to ensure that at least one feature fires for each variable configuration.
            ArrayList<String> biasFeats = new ArrayList<String>();
            biasFeats.add("BIAS_FEATURE");
            
            // Add the bias features.
            FeatureVector fv = new FeatureVector(biasFeats.size() + obsFeats.size());
            FeatureUtils.addFeatures(biasFeats, alphabet, fv, true, prm.featureHashMod);
            
            // Add the other features.
            FeatureUtils.addFeatures(obsFeats, alphabet, fv, false, prm.featureHashMod);
            
            return null;
        }
    };
    
    /**
     * Adds factors and variables to the given factor graph.
     */
    public void build(AnnoSentence sent, ObsFeatureConjoiner ofc, FactorGraph fg, CorpusStatistics cs) {
        // Create tag variables.
        tagVars = new ArrayList<>();
        for (int i=0; i<sent.size(); i++) {
            List<String> stateNames;
            if (prm.posTagVarType == VarType.LATENT) {
                int numLatTags = 10;
                stateNames = new ArrayList<>(numLatTags);
                for (int j=0; j<numLatTags; j++) {
                    stateNames.add("tag"+j);
                }
            } else {
                stateNames = cs.posTagStateNames;
            }
            TagVar v = new TagVar(VarType.PREDICTED, TagVar.getDefaultName(i), stateNames, i);
            tagVars.add(v);
        }
            	
        // Create factors.
        //
        // Unary factor for initial tag.
        TemplateFeatureExtractor fe = new TemplateFeatureExtractor(sent, cs);
        fg.addFactor(new PosTagFactor(new VarSet(tagVars.get(0)), PosTagFactorType.INIT_TAG, 
                ofc, LocalObservations.newPidx(0), fe));
        for (int i=1; i<sent.size(); i++) {
            // Binary factors for subsequent tags.
            VarSet vars = new VarSet(tagVars.get(i-1), tagVars.get(i));
            fg.addFactor(new PosTagFactor(vars, PosTagFactorType.POS_TAG, 
                    ofc, LocalObservations.newPidx(i), fe));
        }
    }
    
    public List<TagVar> getTagVars() {
        return tagVars;
    }
    
}
