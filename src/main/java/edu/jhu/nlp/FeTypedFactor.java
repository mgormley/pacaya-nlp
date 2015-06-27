package edu.jhu.nlp;

import edu.jhu.pacaya.gm.feat.FeatureExtractor;
import edu.jhu.pacaya.gm.model.FeExpFamFactor;
import edu.jhu.pacaya.gm.model.VarSet;

/**
 * A factor which includes its type (i.e. template).
 * 
 * @author mgormley
 */
public class FeTypedFactor extends FeExpFamFactor {

    private static final long serialVersionUID = 1L;

    Enum<?> type;
    
    public FeTypedFactor(VarSet vars, Enum<?> type, FeatureExtractor fe) {
        super(vars, fe);
        this.type = type;
    }
    
    public Enum<?> getFactorType() {
        return type;
    }
    
}