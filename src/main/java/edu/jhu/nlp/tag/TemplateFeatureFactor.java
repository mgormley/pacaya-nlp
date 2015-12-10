package edu.jhu.nlp.tag;

import java.util.ArrayList;
import java.util.List;

import edu.jhu.nlp.features.LocalObservations;
import edu.jhu.nlp.features.TemplateFeatureExtractor;
import edu.jhu.nlp.features.TemplateLanguage.FeatTemplate;
import edu.jhu.nlp.relations.FeatureUtils;
import edu.jhu.pacaya.gm.feat.FactorTemplateList;
import edu.jhu.pacaya.gm.feat.FeatureVector;
import edu.jhu.pacaya.gm.feat.ObsCjExpFamFactor;
import edu.jhu.pacaya.gm.feat.ObsFeatureConjoiner;
import edu.jhu.pacaya.gm.model.VarSet;
import edu.jhu.pacaya.util.FeatureNames;

public class TemplateFeatureFactor extends ObsCjExpFamFactor {

    private static final long serialVersionUID = 1L;
    private LocalObservations local;
    private TemplateFeatureExtractor fe;
    public List<FeatTemplate> templates;
    public int featureHashMod;
    
    public TemplateFeatureFactor(VarSet vars, Object templateKey, ObsFeatureConjoiner ofc, 
            LocalObservations local, TemplateFeatureExtractor fe, 
            List<FeatTemplate> templates, int featureHashMod) {
        super(vars, templateKey, ofc);
        this.local = local;
        this.fe = fe;
        this.templates = templates;
        this.featureHashMod = featureHashMod;
    }
    
    @Override
    public FeatureVector getObsFeatures() {
        FactorTemplateList fts = ofc.getTemplates();
        final FeatureNames alphabet = fts.getTemplate(this).getAlphabet();
        ArrayList<String> obsFeats = new ArrayList<String>();
        fe.addFeatures(templates, local, obsFeats);
        
        // The bias features are used to ensure that at least one feature fires for each variable configuration.
        ArrayList<String> biasFeats = new ArrayList<String>();
        biasFeats.add("BIAS_FEATURE");
        
        // Add the bias features.
        FeatureVector fv = new FeatureVector(biasFeats.size() + obsFeats.size());
        FeatureUtils.addFeatures(biasFeats, alphabet, fv, true, featureHashMod);
        
        // Add the other features.
        FeatureUtils.addFeatures(obsFeats, alphabet, fv, false, featureHashMod);
        
        return fv;
    }
}