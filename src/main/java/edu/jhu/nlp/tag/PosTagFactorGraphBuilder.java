package edu.jhu.nlp.tag;

import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.primitives.Bytes;

import edu.jhu.nlp.CorpusStatistics;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.data.simple.IntAnnoSentence;
import edu.jhu.nlp.features.LocalObservations;
import edu.jhu.nlp.features.TemplateFeatureExtractor;
import edu.jhu.nlp.features.TemplateLanguage.FeatTemplate;
import edu.jhu.nlp.features.TemplateSets;
import edu.jhu.pacaya.gm.feat.FeatureVector;
import edu.jhu.pacaya.gm.feat.ObsFeatureConjoiner;
import edu.jhu.pacaya.gm.model.FactorGraph;
import edu.jhu.pacaya.gm.model.Var;
import edu.jhu.pacaya.gm.model.Var.VarType;
import edu.jhu.pacaya.gm.model.VarConfig;
import edu.jhu.pacaya.gm.model.VarSet;
import edu.jhu.pacaya.util.Prm;
import edu.jhu.pacaya.util.collections.QLists;
import edu.jhu.prim.Primitives;
import edu.jhu.prim.util.SafeCast;

public class PosTagFactorGraphBuilder {

    private static final Logger log = LoggerFactory.getLogger(PosTagFactorGraphBuilder.class);

    public static class PosTagFactorGraphBuilderPrm extends Prm {
        private static final long serialVersionUID = 1L;
        /** The type of the tag variables. */
        public VarType posTagVarType = VarType.LATENT;
        /** Whether to use fast features only. */
        public boolean onlyFast = true;
        /** Feature templates. */
        public List<FeatTemplate> templates = TemplateSets.getFromResource(TemplateSets.custom2TagFeatsResource);
        /** The value of the mod for use in the feature hashing trick. If <= 0, feature-hashing will be disabled. */
        public int featureHashMod = 1000000;
        /** Whether to include unary factors on each tag. */
        public boolean unigramFactors = true;
        /** Whether to include binary factors on each pair of tags. */
        public boolean bigramFactors = true;
    }
    
    public enum PosTagFactorType {
        TAG_BIGRAM, INIT_TAG, TAG_UNIGRAM
    }
        
    private PosTagFactorGraphBuilderPrm prm;
    private List<Var> tagVars;
    
    public PosTagFactorGraphBuilder(PosTagFactorGraphBuilderPrm prm) {
        this.prm = prm;
    }
    
    /**
     * Adds factors and variables to the given factor graph.
     */
    public void build(IntAnnoSentence isent, ObsFeatureConjoiner ofc, FactorGraph fg, CorpusStatistics cs) {
        ofc.takeNoteOfFeatureHashMod(prm.featureHashMod);
        
        // Create tag variables.
        tagVars = new ArrayList<>();
        for (int i=0; i<isent.size(); i++) {
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
            Var v = new Var(prm.posTagVarType, stateNames.size(), "tag"+i, stateNames);
            tagVars.add(v);
        }

        if (prm.onlyFast && prm.featureHashMod > 0) {
            addFastFactors(isent, fg);
        } else {
            addSlowFactors(isent.getAnnoSentence(), ofc, fg, cs);
        }
    }

    private void addFastFactors(IntAnnoSentence isent, FactorGraph fg) {
	// TODO: Add unary INIT_TAG factor as in addSlowFactors.
        for (int i=0; i<isent.size(); i++) {
            if (prm.unigramFactors) {
                // Unary factor for each tag.
                VarSet vars = new VarSet(tagVars.get(i));
                final FeatureVector obsFeats = new FeatureVector();
                BitshiftTokenFeatures.addUnigramFeatures(isent, i, obsFeats, -1, (short) 0); // config=0
                fg.addFactor(new HashObsFeatsFactor(vars, obsFeats, 
                        SafeCast.safeIntToShort(PosTagFactorType.TAG_UNIGRAM.ordinal()), prm.featureHashMod));
            }
            if (i > 0 && prm.bigramFactors) {
                // Binary factor for each pair of tags.
                VarSet vars = new VarSet(tagVars.get(i-1), tagVars.get(i));
                final FeatureVector obsFeats = new FeatureVector();
                BitshiftTokenFeatures.addBigramFeatures(isent, i, obsFeats, -1, (short) 0); // config=0
                fg.addFactor(new HashObsFeatsFactor(vars, obsFeats, 
                        SafeCast.safeIntToShort(PosTagFactorType.TAG_UNIGRAM.ordinal()), prm.featureHashMod));
            }
        }
    }
    
    protected void addSlowFactors(AnnoSentence sent, ObsFeatureConjoiner ofc, FactorGraph fg, CorpusStatistics cs) {
        // Features for tag bigrams.
        List<FeatTemplate> templates = QLists.getList(); // Only use the bias feature.
        
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
    
    public List<Var> getTagVars() {
        return tagVars;
    }

    /* ------------------------- Encode ------------------------- */
    public void addVarAssignments(List<String> tags, VarConfig vc) {
        for (int i=0; i<tagVars.size(); i++) {
            Var var = tagVars.get(i);
            vc.put(var, tags.get(i));
        }
    }
    
    /* ------------------------- Decode ------------------------- */
    public List<String> getTagsFromMbrVarConfig(VarConfig mbrVarConfig) {
        ArrayList<String> tags = new ArrayList<>();
        for (int i=0; i<tagVars.size(); i++) {
            Var var = tagVars.get(i);
            tags.add(mbrVarConfig.getStateName(var));
        }
        return tags;
    }
    
}
