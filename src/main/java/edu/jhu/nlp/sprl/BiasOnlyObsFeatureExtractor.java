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
        FeatureVector fv = new FeatureVector();

        ObsFeTypedFactor f = (ObsFeTypedFactor) factor;
        
        int templateId = ofc.getTemplates().getTemplateId(f);
        // if we never saw this in training, we will ignore it
        if (templateId >= 0) {
            FeatureNames alphabet = ofc.getTemplates().getTemplate(f).getAlphabet();
            List<String> biasFeats = new ArrayList<String>();
            biasFeats.add("BIAS_FEATURE");
            FeatureUtils.addFeatures(biasFeats, alphabet, fv, true, featureHashMod);
        }
        return fv;
    }


}
