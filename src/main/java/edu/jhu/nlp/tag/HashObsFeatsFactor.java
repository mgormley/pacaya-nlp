package edu.jhu.nlp.tag;

import edu.jhu.nlp.features.BitPacking;
import edu.jhu.pacaya.gm.feat.FeatureVector;
import edu.jhu.pacaya.gm.model.ExpFamFactor;
import edu.jhu.pacaya.gm.model.VarSet;
import edu.jhu.prim.util.SafeCast;

/**
 * Highly efficient factor for using observed features. This factor hashes the observation property
 * vector g(x), a factor type, and a (predicted) variable configuration. The result is a complete
 * feature vector f(x,y) which will explicitly share parameters only with factors that have the same
 * factor type (i.e. template). Of course, since the features are created by hashing, collisions are
 * possible.
 * 
 * @author mgormley
 */
public class HashObsFeatsFactor extends ExpFamFactor {

    private static final long serialVersionUID = 1L;
    private FeatureVector obsFeats;
    private int featureHashMod;
    private short factorType;

    /**
     * 
     * @param vars The variables for this factor.
     * @param obsFeats The properties of the observations.
     * @param factorType The factor template. Only factors with the same factorType will explicitly
     *            share parameters. It is assumed, but NOT checked that all factors with the same factor 
     *            type will have the same variable arrangement: that is, if the config=1 on one factor it 
     *            should correspond to the same config on the factor with which parameters are shared.
     * @param featureHashMod The feature hash mod.
     */
    public HashObsFeatsFactor(VarSet vars, FeatureVector obsFeats, short factorType, int featureHashMod) {
        super(vars);
        this.obsFeats = obsFeats;
        this.featureHashMod = featureHashMod;
        this.factorType = factorType;        
    }
    
    @Override
    public FeatureVector getFeatures(int config) {
        short shortConfig = SafeCast.safeIntToShort(config);
        int[] idxs = obsFeats.getInternalIndices();
        int used = obsFeats.getUsed();
        FeatureVector feats = new FeatureVector(obsFeats.getUsed());
        for (int k=0; k<used; k++) {
            long feat = BitPacking.encodeFeatureISS_(idxs[k], shortConfig, factorType);
            BitshiftTokenFeatures.addFeat(feats, featureHashMod, feat);
        }
        return feats;
    }
    
}