package edu.jhu.nlp.tag;

import edu.jhu.nlp.features.BitPacking;
import edu.jhu.pacaya.gm.feat.FeatureVector;
import edu.jhu.pacaya.gm.model.ExpFamFactor;
import edu.jhu.pacaya.gm.model.VarSet;

public class HashObsFeatsFactor extends ExpFamFactor {

    private static final long serialVersionUID = 1L;
    private FeatureVector obsFeats;
    private int featureHashMod;

    public HashObsFeatsFactor(VarSet vars, FeatureVector obsFeats, int featureHashMod) {
        super(vars);
        this.obsFeats = obsFeats;
        this.featureHashMod = featureHashMod;
    }
    
    @Override
    public FeatureVector getFeatures(int config) {
        // TODO: Double check that magic is not a bug - should config be opened up?
        // TODO: Currently, there is not a factor type associated with this.
        int[] idxs = obsFeats.getInternalIndices();
        int used = obsFeats.getUsed();
        FeatureVector feats = new FeatureVector(obsFeats.getUsed());
        for (int k=0; k<used; k++) {
            long feat = BitPacking.encodeFeatureII__(config, idxs[k]);
            BitshiftTokenFeatures.addFeat(feats, featureHashMod, feat);
        }
        return feats;
    }
    
}