package edu.jhu.nlp.fcm;

import java.util.List;

import edu.jhu.nlp.data.simple.IntAnnoSentence;
import edu.jhu.nlp.embed.Embeddings;
import edu.jhu.pacaya.autodiff.AbstractModule;
import edu.jhu.pacaya.autodiff.MVec;
import edu.jhu.pacaya.autodiff.Module;
import edu.jhu.pacaya.autodiff.Tensor;
import edu.jhu.pacaya.autodiff.erma.MVecFgModel;
import edu.jhu.pacaya.gm.feat.FeatureVector;
import edu.jhu.pacaya.gm.model.Var;
import edu.jhu.pacaya.gm.model.VarSet;
import edu.jhu.pacaya.gm.model.VarTensor;
import edu.jhu.pacaya.util.collections.QLists;
import edu.jhu.pacaya.util.semiring.Algebra;
import edu.jhu.prim.util.Lambda.FnIntDoubleToVoid;

public class FcmModule extends AbstractModule<VarTensor> implements Module<VarTensor> {

    private IntAnnoSentence isent;
    private Embeddings embeddings;
    private Module<MVecFgModel> modIn;
    private Var var;
    private boolean fineTuning;
    
    public FcmModule(Module<MVecFgModel> modIn, Algebra s) {
        super(s);
        this.modIn = modIn;
    }

    @Override
    public VarTensor forward() {
        Tensor param = null; // TODO: get from a module
        Tensor embed = null; // TODO: get from a module
        int embedDim = -1; // TODO:
        int numFeats = -1; // TODO:
        
        VarTensor fac = new VarTensor(s, new VarSet(var));
        for (int i=0; i<isent.size(); i++) {
            int w_i = isent.getWord(i);
            FeatureVector f_i = getFeatures(i);
            for (int y=0; y<fac.size(); y++) {
                for (int m=0; m<embedDim; m++) {
                    // Sparse loop over features.
                    for(int j=0; j<f_i.getUsed(); j++) {
                        int k = f_i.getInternalIndices()[j];
                        double f_ik = f_i.getInternalValues()[j];
                        assert 0 <= k && k < numFeats;
                        // Add to the factor score.
                        double score = param.get(y, m, k) * embed.get(w_i, m) * f_ik;
                        fac.add(score, y);
                    }
                }
            }
        }
        return fac;
    }

    @Override
    public void backward() {
        Tensor param = null; // TODO: get from a module
        Tensor embed = null; // TODO: get from a module
        Tensor paramAdj = null; // TODO: Get from a module.
        Tensor embedAdj = null; // TODO: Get from a module.
        int embedDim = -1; // TODO:
        int numFeats = -1; // TODO:
        
        // Backprop to tensor parameters.
//        for (int i=0; i<isent.size(); i++) {
//            FeatureVector f_i = getFeatures(i);
//            for (int y=0; y<fac.size(); y++) {
//                for (int m=0; m<embedDim; m++) {
//                    // Sparse loop over features.
//                    for(int j=0; j<f_i.getUsed(); j++) {
//                        
//                    }
//                }
//            }
//        }
        
        if (fineTuning) {
            // Backprop to embedding parameters.
            
        }
    }

    @Override
    public List<? extends Module<? extends MVec>> getInputs() {
        return QLists.getList(modIn);
    }

    private FeatureVector getFeatures(int i) {
        // TODO Auto-generated method stub
        return null;
    }

}
