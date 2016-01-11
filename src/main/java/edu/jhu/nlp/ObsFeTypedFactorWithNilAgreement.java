package edu.jhu.nlp;

import java.util.List;

import edu.jhu.pacaya.gm.feat.ObsFeatureConjoiner;
import edu.jhu.pacaya.gm.feat.ObsFeatureExtractor;
import edu.jhu.pacaya.gm.model.FgModel;
import edu.jhu.pacaya.gm.model.Var;
import edu.jhu.pacaya.gm.model.VarConfig;
import edu.jhu.pacaya.gm.model.VarSet;

/**
 * puts non-zero probability mass on only a given set of configurations
 * 
 */
public class ObsFeTypedFactorWithNilAgreement extends ObsFeTypedFactor {
    private static final long serialVersionUID = -2667403310901430389L;
    private List<Var> vars;
    private List<Integer> nilStates;

    /**
     * makes a VarSet from a list of vars (done here so that super can be called
     * in the first line below)
     */
    private static VarSet makeVarSet(List<Var> vars) {
        VarSet vs = new VarSet();
        vs.addAll(vars);
        return vs;
    }

    public ObsFeTypedFactorWithNilAgreement(List<Var> vars, List<Integer> nilStates, Enum<?> type, Object templateKey,
            ObsFeatureConjoiner ofc, ObsFeatureExtractor obsFe) {
        super(makeVarSet(vars), type, templateKey, ofc, obsFe);
        this.vars = vars;
        this.nilStates = nilStates;
    }

    /**
     * @return true if the config corresponds to a setting of the varset where
     *         either all vars are set to nil or none are set to nil
     */
    public static boolean allOrNoneNil(VarSet vSet, List<Var> vs, List<Integer> nils, int config) {
        VarConfig vc = vSet.getVarConfig(config);
        // exactly one of these must always be true
        boolean atLeastOneIsNil = false; // so far none are nill
        boolean allAreNil = true; // and all are nil
        for (int i = 0; i < vs.size(); i++) {
            boolean isNil = (vc.getState(vs.get(i)) == nils.get(i));
            if (isNil) {
                atLeastOneIsNil = true;
            } else {
                allAreNil = false;
            }
            boolean exactlyOne = (atLeastOneIsNil || allAreNil) && !(atLeastOneIsNil && allAreNil);
            if (!exactlyOne) {
                return false;
            }
        }
        return true;

    }

    @Override
    protected double getDotProd(int config, FgModel model) {
        // walk down the variables and make sure that either all or none are set
        // to the respective nil state
        if (allOrNoneNil(getVars(), vars, nilStates, config)) {
            return super.getDotProd(config, model);
        } else {
            //return -10.0;
            //return 0.0;
            return Double.NEGATIVE_INFINITY;
        }

    }

}