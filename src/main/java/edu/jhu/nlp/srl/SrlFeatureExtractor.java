package edu.jhu.nlp.srl;

import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.CorpusStatistics;
import edu.jhu.nlp.ObsFeTypedFactor;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.depparse.DepParseFactorGraphBuilder.DepParseFactorTemplate;
import edu.jhu.nlp.features.FeaturizedSentence;
import edu.jhu.nlp.features.LocalObservations;
import edu.jhu.nlp.features.TemplateFeatureExtractor;
import edu.jhu.nlp.features.TemplateLanguage.FeatTemplate;
import edu.jhu.nlp.features.TemplateSets;
import edu.jhu.nlp.joint.JointNlpFactorGraph.JointFactorTemplate;
import edu.jhu.nlp.relations.FeatureUtils;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.RoleVar;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.SenseVar;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.SrlFactorTemplate;
import edu.jhu.pacaya.gm.data.UFgExample;
import edu.jhu.pacaya.gm.feat.FactorTemplateList;
import edu.jhu.pacaya.gm.feat.FeatureVector;
import edu.jhu.pacaya.gm.feat.ObsFeExpFamFactor;
import edu.jhu.pacaya.gm.feat.ObsFeatureExtractor;
import edu.jhu.pacaya.gm.model.Var;
import edu.jhu.pacaya.gm.model.VarSet;
import edu.jhu.pacaya.gm.model.globalfac.LinkVar;
import edu.jhu.pacaya.util.FeatureNames;
import edu.jhu.pacaya.util.Prm;

/**
 * Feature extractor for SRL. All the "real" feature extraction is done in
 * SentFeatureExtraction which considers only the observations.
 * 
 * @author mgormley
 * @author mmitchell
 */
public class SrlFeatureExtractor implements ObsFeatureExtractor {

    public static class SrlFeatureExtractorPrm extends Prm {
        private static final long serialVersionUID = 1L;
        /** For testing only: this will ensure that the only feature returned is the bias feature. */
        public boolean biasOnly = false;
        /** Whether to use ONLY feature templates. */
        public boolean useTemplates = true;
        /** Feature templates. */
        public List<FeatTemplate> senseTemplates = TemplateSets.getNaradowskySenseUnigramFeatureTemplates();
        public List<FeatTemplate> argTemplates = TemplateSets.getNaradowskyArgUnigramFeatureTemplates();
        /** The value of the mod for use in the feature hashing trick. If <= 0, feature-hashing will be disabled. */
        public int featureHashMod = -1;
        /** Whether to create human interpretable feature names when possible. */
        public boolean humanReadable = true;
    }
    
    private static final Logger log = LoggerFactory.getLogger(SrlFeatureExtractor.class); 
    
    private SrlFeatureExtractorPrm prm;
    private FactorTemplateList fts;
    private TemplateFeatureExtractor ext;
    
    public SrlFeatureExtractor(SrlFeatureExtractorPrm prm, AnnoSentence sent, CorpusStatistics cs, FactorTemplateList fts) {
        this.prm = prm;
        if (prm.useTemplates) {
            FeaturizedSentence fSent = new FeaturizedSentence(sent, cs);
            ext = new TemplateFeatureExtractor(fSent, cs);
        }
        this.fts = fts;
    }

    // For testing only.
    void init(FactorTemplateList fts) {
        this.fts = fts;
    }
    
    @Override
    public FeatureVector calcObsFeatureVector(ObsFeExpFamFactor factor) {
        ObsFeTypedFactor f = (ObsFeTypedFactor) factor;
        Enum<?> ft = f.getFactorType();
        VarSet vars = f.getVars();
        
        // Get the observation features.
        ArrayList<String> obsFeats;
        FeatureNames alphabet;
        if (ft == JointFactorTemplate.LINK_ROLE_BINARY || ft == DepParseFactorTemplate.UNARY 
                || ft == SrlFactorTemplate.ROLE_UNARY || ft == SrlFactorTemplate.SENSE_ROLE_BINARY
                || ft == JointFactorTemplate.ROLE_C_TAG_BINARY || ft == JointFactorTemplate.ROLE_P_TAG_BINARY) {
            // Look at the variables to determine the parent and child.
            int parent = Integer.MIN_VALUE;
            int child = Integer.MIN_VALUE;
            for (int i=0; i<vars.size(); i++) {
                Var var = vars.get(i);
                if (var instanceof LinkVar) {
                    parent = ((LinkVar)var).getParent();
                    child = ((LinkVar)var).getChild();
                } else if (var instanceof RoleVar) {
                    parent = ((RoleVar)var).getParent();
                    child = ((RoleVar)var).getChild();
                }
            }
            if (parent == Integer.MIN_VALUE && child == Integer.MIN_VALUE) {
                throw new RuntimeException("Unknown variable type for ft: " + ft);
            }

            // Get features on the observations for a pair of words.
            // IMPORTANT NOTE: We include the case where the parent is the Wall node (position -1).
            obsFeats = createFeatureSet(parent, child);
        } else if (ft == SrlFactorTemplate.SENSE_UNARY) {
            SenseVar var = (SenseVar) vars.iterator().next();
            int parent = var.getParent();
            obsFeats = createFeatureSet(parent);
        } else {
            throw new RuntimeException("Unsupported template: " + ft);
        }
        alphabet = fts.getTemplate(f).getAlphabet();
                
        if (log.isTraceEnabled()) {
            log.trace("Num obs features in factor: " + obsFeats.size());
        }
                
        // The bias features are used to ensure that at least one feature fires for each variable configuration.
        ArrayList<String> biasFeats = new ArrayList<String>();
        biasFeats.add("BIAS_FEATURE");
        
        // Add the bias features.
        FeatureVector fv = new FeatureVector(biasFeats.size() + obsFeats.size());
        FeatureUtils.addFeatures(biasFeats, alphabet, fv, true, prm.featureHashMod);
        
        // Add the other features.
        FeatureUtils.addFeatures(obsFeats, alphabet, fv, false, prm.featureHashMod);
        
        return fv;
    }
    

    // ----------------- Extracting Features on the Observations ONLY -----------------

    /**
     * Creates a feature set for the given word position.
     * 
     * This defines a feature function of the form f(x, i), where x is a vector
     * representing all the observations about a sentence and i is a position in
     * the sentence.
     * 
     * Examples where this feature function would be used include a unary factor
     * on a Sense variable in SRL, or a syntactic Link variable where the parent
     * is the "Wall" node.
     * 
     * @param idx The position of a word in the sentence.
     * @return The features.
     */
    public ArrayList<String> createFeatureSet(int idx) {
        ArrayList<String> feats = new ArrayList<String>();
        if (prm.biasOnly) { return feats; }
        if (prm.useTemplates) {
            addTemplateSoloFeatures(idx, feats);
            return feats;
        }
        return feats;
    }
    
    /**
     * Creates a feature set for the given pair of word positions.
     * 
     * This defines a feature function of the form f(x, i, j), where x is a
     * vector representing all the observations about a sentence and i and j are
     * positions in the sentence.
     * 
     * Examples where this feature function would be used include factors
     * including a Role or Link variable in the SRL model, where both the parent
     * and child are tokens in the sentence.
     * 
     * @param pidx The "parent" position.
     * @param aidx The "child" position.
     * @return The features.
     */
    public ArrayList<String> createFeatureSet(int pidx, int aidx) {
        ArrayList<String> feats = new ArrayList<String>();
        if (prm.biasOnly) { return feats; }
        if (prm.useTemplates) {
            addTemplatePairFeatures(pidx, aidx, feats);
            return feats;
        }
        return feats;
    }

    private void addTemplateSoloFeatures(int idx, ArrayList<String> feats) {
        if (prm.senseTemplates == null) {
            throw new IllegalStateException("Solo template set must be specified");
        }
        ext.addFeatures(prm.senseTemplates, LocalObservations.newPidx(idx), feats);
    }

    private void addTemplatePairFeatures(int pidx, int aidx, ArrayList<String> feats) {
        if (prm.argTemplates == null) {
            throw new IllegalStateException("Pair template set must be specified");
        }
        ext.addFeatures(prm.argTemplates, LocalObservations.newPidxCidx(pidx, aidx), feats);        
    }
    
}
