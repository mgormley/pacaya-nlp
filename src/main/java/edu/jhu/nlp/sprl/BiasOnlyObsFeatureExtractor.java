package edu.jhu.nlp.sprl;

import java.util.ArrayList;
import java.util.List;

import edu.jhu.nlp.ObsFeTypedFactor;
import edu.jhu.nlp.relations.FeatureUtils;
import edu.jhu.pacaya.gm.feat.FeatureVector;
import edu.jhu.pacaya.gm.feat.ObsFeExpFamFactor;
import edu.jhu.pacaya.gm.feat.ObsFeatureConjoiner;
import edu.jhu.pacaya.gm.feat.ObsFeatureExtractor;
import edu.jhu.pacaya.util.FeatureNames;

/**
 * "Feature Extractor" that doesn't extract any features (useful if only the bias is wanted)
 *
 */
public class BiasOnlyObsFeatureExtractor implements ObsFeatureExtractor {

    private ObsFeatureConjoiner ofc;
    private int featureHashMod;
    
    public BiasOnlyObsFeatureExtractor(ObsFeatureConjoiner ofc, int featureHashMod) {
        this.ofc = ofc;
        this.featureHashMod = featureHashMod;
    }
    
    @Override
    public FeatureVector calcObsFeatureVector(ObsFeExpFamFactor factor) {
        ObsFeTypedFactor f = (ObsFeTypedFactor) factor;
        
        FeatureNames alphabet = ofc.getTemplates().getTemplate(f).getAlphabet();
        List<String> biasFeats = new ArrayList<String>();
        biasFeats.add("BIAS_FEATURE");

        // Add the bias features.
        FeatureVector fv = new FeatureVector(biasFeats.size());
        FeatureUtils.addFeatures(biasFeats, alphabet, fv, true, featureHashMod);

        return fv;
    }


}
