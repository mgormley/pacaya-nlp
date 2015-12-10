package edu.jhu.nlp.fcm;

import java.util.ArrayList;
import java.util.List;

import edu.jhu.pacaya.gm.feat.FeatureVector;
import edu.jhu.pacaya.gm.model.VarSet;
import edu.jhu.pacaya.util.FeatureNames;

public interface WordFeatures {

    /** Gets a length n list of feature vectors, one for each word in a length n sentence. */
    public List<FeatureVector> getFeatures(VarSet vars);

    /** Gets the alphabet for the feature vectors. */
    public FeatureNames getAlphabet();

    default List<FeatureVector> getListOfEmptyFvs(int n) {
        List<FeatureVector> fvs = new ArrayList<>(n);
        for (int i=0; i<n; i++) {
            fvs.add(new FeatureVector());
        }
        return fvs;
    }
    
}
