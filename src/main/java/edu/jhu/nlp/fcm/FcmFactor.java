package edu.jhu.nlp.fcm;

import java.util.List;

import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.embed.Embeddings;
import edu.jhu.nlp.relations.RelationsFactorGraphBuilder.RelVar;
import edu.jhu.pacaya.autodiff.Module;
import edu.jhu.pacaya.gm.feat.FeatureVector;
import edu.jhu.pacaya.gm.feat.ObsFeatureConjoiner;
import edu.jhu.pacaya.gm.model.AutodiffFactor;
import edu.jhu.pacaya.gm.model.ExplicitFactor;
import edu.jhu.pacaya.gm.model.Factor;
import edu.jhu.pacaya.gm.model.FgModel;
import edu.jhu.pacaya.gm.model.FgModelIdentity;
import edu.jhu.pacaya.gm.model.IFgModel;
import edu.jhu.pacaya.gm.model.MVecFgModel;
import edu.jhu.pacaya.gm.model.VarSet;
import edu.jhu.pacaya.gm.model.VarTensor;
import edu.jhu.pacaya.util.semiring.Algebra;
import edu.jhu.pacaya.util.semiring.LogSemiring;

public class FcmFactor extends ExplicitFactor implements Factor, AutodiffFactor {

    private static final long serialVersionUID = 1L;
    private AnnoSentence sent;
    private ObsFeatureConjoiner ofc;
    private Embeddings embeddings;
    private boolean fineTuning;
    private WordFeatures wf;
    // This will be cached.
    private List<FeatureVector> wordFeats;

    public FcmFactor(VarSet vars, AnnoSentence sent, Embeddings embeddings, ObsFeatureConjoiner ofc, boolean fineTuning, WordFeatures wf) {
        super(vars);
        this.sent = sent;
        this.embeddings = embeddings;
        this.ofc = ofc;
        this.fineTuning = fineTuning;
        this.wf = wf;
    }

    @Override
    public void updateFromModel(FgModel model) {
        // Set the values on the ExplicitFactor. This is done automatically by the call to forward().
        getFactorModule(new FgModelIdentity(model), LogSemiring.getInstance()).forward();
    }

    @Override
    public void addExpectedPartials(IFgModel counts, VarTensor factorMarginal, double multiplier) {
        if (multiplier != 0) {
            throw new RuntimeException("addExpectedPartials is only implemented for feature counting");
        }
        // Do feature extraction to populate the alphabet.
        FcmModule fcm = getFactorModule(null, s);
        ofc.requestReserved(fcm.getNumParams());
    }
    
    @Override
    public FcmModule getFactorModule(Module<MVecFgModel> modIn, Algebra s) {
        if (wordFeats == null) {
            wordFeats = wf.getFeatures(getVars());
        }
        return new FcmModule(modIn, s, wordFeats, wf.getAlphabet(), 
                getVars(), sent, embeddings, ofc.getReservedOffset(), fineTuning,
                this);
    }

}
