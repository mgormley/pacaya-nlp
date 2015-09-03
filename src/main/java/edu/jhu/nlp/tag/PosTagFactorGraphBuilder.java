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
import edu.jhu.pacaya.gm.feat.ObsFeatureConjoiner;
import edu.jhu.pacaya.gm.model.FactorGraph;
import edu.jhu.pacaya.gm.model.Var;
import edu.jhu.pacaya.gm.model.Var.VarType;
import edu.jhu.pacaya.gm.model.VarSet;
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
        /** Whether to include unary factors on each tag. */
        public boolean unigramFactors = true;
        /** Whether to include binary factors on each pair of tags. */
        public boolean bigramFactors = true;
    }
    
    public enum PosTagFactorType {
        TAG_BIGRAM, INIT_TAG, TAG_UNIGRAM
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
            TagVar v = new TagVar(prm.posTagVarType, TagVar.getDefaultName(i), stateNames, i);
            tagVars.add(v);
        }
        
        // Create factors.
        TemplateFeatureExtractor fe = new TemplateFeatureExtractor(sent, cs);
        if (prm.bigramFactors) {
            // Unary factor for initial tag, since we can't do a bigram factor for it.
            VarSet vars = new VarSet(tagVars.get(0));
            fg.addFactor(new TemplateFeatureFactor(vars, PosTagFactorType.INIT_TAG, 
                    ofc, LocalObservations.newPidx(0), fe,
                    prm.templates, prm.featureHashMod));
        }
        for (int i=0; i<sent.size(); i++) {
            if (prm.unigramFactors) {
                // Unary factor for each tag.
                VarSet vars = new VarSet(tagVars.get(i));
                fg.addFactor(new TemplateFeatureFactor(vars, PosTagFactorType.TAG_UNIGRAM, 
                        ofc, LocalObservations.newPidx(i), fe,
                        prm.templates, prm.featureHashMod));
            }
            if (i > 0 && prm.bigramFactors) {
                // Binary factor for each pair of tags.
                VarSet vars = new VarSet(tagVars.get(i-1), tagVars.get(i));
                fg.addFactor(new TemplateFeatureFactor(vars, PosTagFactorType.TAG_BIGRAM,
                        ofc, LocalObservations.newPidx(i), fe,
                        prm.templates, prm.featureHashMod));
            }
        }
    }
    
    public List<TagVar> getTagVars() {
        return tagVars;
    }
    
}
