package edu.jhu.nlp.joint;

import edu.jhu.nlp.CorpusStatistics;
import edu.jhu.nlp.joint.JointNlpFactorGraph.JointNlpFactorGraphPrm;
import edu.jhu.pacaya.gm.feat.ObsFeatureConjoiner;
import edu.jhu.pacaya.gm.model.FgModel;

public class JointNlpFgModel extends FgModel {

    private static final long serialVersionUID = 5827437917567173421L;
    private CorpusStatistics cs;
    private JointNlpFactorGraphPrm fgPrm;
    private ObsFeatureConjoiner ofc;
    
    public JointNlpFgModel(CorpusStatistics cs, ObsFeatureConjoiner ofc, JointNlpFactorGraphPrm fgPrm) {
        super(ofc.getNumParams(), ofc.getParamNames());
        this.cs = cs;
        this.ofc = ofc;
        this.fgPrm = fgPrm;
    }

    public CorpusStatistics getCs() {
        return cs;
    }

    public JointNlpFactorGraphPrm getFgPrm() {
        return fgPrm;
    }
    
    public ObsFeatureConjoiner getOfc() {
        return ofc;
    }
    
}
