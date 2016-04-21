package edu.jhu.nlp.sprl;

import edu.jhu.pacaya.gm.feat.FeatureVector;
import edu.jhu.pacaya.gm.feat.ObsFeExpFamFactor;
import edu.jhu.pacaya.gm.feat.ObsFeatureExtractor;

/**
 * "Feature Extractor" that doesn't extract any features (useful if only the bias is wanted)
 *
 */
public class BiasOnlyObsFeatureExtractor implements ObsFeatureExtractor {
    private static BiasOnlyObsFeatureExtractor instance;

    @Override
    public FeatureVector calcObsFeatureVector(ObsFeExpFamFactor factor) {
        return new FeatureVector();
    }

    /**
     * Access to a global static instance
     */
    public static BiasOnlyObsFeatureExtractor instance() {
        if (instance == null) {
            instance = new BiasOnlyObsFeatureExtractor();
        }
        return instance;
    }

}
