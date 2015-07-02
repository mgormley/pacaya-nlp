package edu.jhu.nlp.fcm;

import edu.jhu.pacaya.autodiff.Module;
import edu.jhu.pacaya.autodiff.erma.AutodiffFactor;
import edu.jhu.pacaya.autodiff.erma.MVecFgModel;
import edu.jhu.pacaya.gm.model.Factor;
import edu.jhu.pacaya.gm.model.FgModel;
import edu.jhu.pacaya.gm.model.IFgModel;
import edu.jhu.pacaya.gm.model.VarConfig;
import edu.jhu.pacaya.gm.model.VarSet;
import edu.jhu.pacaya.gm.model.VarTensor;
import edu.jhu.pacaya.util.semiring.Algebra;

public class FcmFactor implements Factor, AutodiffFactor {


    @Override
    public VarSet getVars() {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public void updateFromModel(FgModel model) {
        // TODO Auto-generated method stub
        
    }

    @Override
    public double getLogUnormalizedScore(VarConfig goldConfig) {
        // TODO Auto-generated method stub
        return 0;
    }

    @Override
    public double getLogUnormalizedScore(int goldConfig) {
        // TODO Auto-generated method stub
        return 0;
    }

    @Override
    public void addExpectedPartials(IFgModel counts, VarTensor factorMarginal, double multiplier) {
        // TODO Auto-generated method stub
        
    }

    @Override
    public int getId() {
        // TODO Auto-generated method stub
        return 0;
    }

    @Override
    public void setId(int id) {
        // TODO Auto-generated method stub
        
    }
    
    @Override
    public Module<?> getFactorModule(Module<MVecFgModel> modIn, Algebra s) {
        // TODO Auto-generated method stub
        return null;
    }

}
