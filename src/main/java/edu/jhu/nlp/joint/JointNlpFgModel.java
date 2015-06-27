package edu.jhu.nlp.joint;

import edu.jhu.nlp.CorpusStatistics;
import edu.jhu.nlp.joint.JointNlpEncoder.JointNlpFeatureExtractorPrm;
import edu.jhu.pacaya.gm.feat.ObsFeatureConjoiner;
import edu.jhu.pacaya.gm.model.FgModel;

public class JointNlpFgModel extends FgModel {

    private static final long serialVersionUID = 5827437917567173421L;
    private CorpusStatistics cs;
    private JointNlpFeatureExtractorPrm fePrm;
    private ObsFeatureConjoiner ofc;
    
    public JointNlpFgModel(CorpusStatistics cs, ObsFeatureConjoiner ofc, JointNlpFeatureExtractorPrm fePrm) {
        super(ofc.getNumParams(), ofc.getParamNames());
        this.cs = cs;
        this.ofc = ofc;
        this.fePrm = fePrm;
    }

    public CorpusStatistics getCs() {
        return cs;
    }

    public JointNlpFeatureExtractorPrm getFePrm() {
        return fePrm;
    }
    
    public ObsFeatureConjoiner getOfc() {
        return ofc;
    }
    
}
