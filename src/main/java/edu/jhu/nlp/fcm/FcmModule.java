package edu.jhu.nlp.fcm;

import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.data.simple.IntAnnoSentence;
import edu.jhu.nlp.embed.Embeddings;
import edu.jhu.pacaya.autodiff.AbstractModule;
import edu.jhu.pacaya.autodiff.MVec;
import edu.jhu.pacaya.autodiff.Module;
import edu.jhu.pacaya.autodiff.Tensor;
import edu.jhu.pacaya.autodiff.VTensor;
import edu.jhu.pacaya.gm.feat.FeatureVector;
import edu.jhu.pacaya.gm.feat.ObsFeatureConjoiner;
import edu.jhu.pacaya.gm.model.FgModel;
import edu.jhu.pacaya.gm.model.MVecFgModel;
import edu.jhu.pacaya.gm.model.Var;
import edu.jhu.pacaya.gm.model.VarSet;
import edu.jhu.pacaya.gm.model.VarTensor;
import edu.jhu.pacaya.util.FeatureNames;
import edu.jhu.pacaya.util.collections.QLists;
import edu.jhu.pacaya.util.semiring.Algebra;
import edu.jhu.pacaya.util.semiring.RealAlgebra;
import edu.jhu.prim.util.math.FastMath;
import edu.jhu.prim.vector.IntDoubleVector;

/**
 * Computes the score given by the FCM without exponentiating.
 * 
 * The input parameters consist of: 
 * (1) The FCM model parameters, T, a KxDxY Tensor, 
 * (2) The word embeddings, e, a VxD Tensor.
 * 
 * The output values are a vector (length Y) of scores (one per label). 
 * These are intended to be the log-scores for a factor.
 * 
 * @author mgormley
 */
public class FcmModule extends AbstractModule<VarTensor> implements Module<VarTensor> {
    
    private static final Logger log = LoggerFactory.getLogger(FcmModule.class);

    private Module<MVecFgModel> modIn;
    private List<FeatureVector> feats;
    private AnnoSentence sent;
    private Embeddings embeddings;
    private VarSet vars;
    private boolean fineTuning;
    
    private VarTensor scores;
    // Model dimensions.
    private final int numFeats;
    private final int embedDim;
    private final int numWordTypes;
    private final int numLabels;
    // Model parameter offset.
    private final int embedOffset;
    private final int tparamOffset;
    
    private VarTensor toUpdate; // TODO: Remove this hack.
    
    public FcmModule(Module<MVecFgModel> modIn, Algebra s, List<FeatureVector> feats, FeatureNames featAlphabet, 
            VarSet vars, AnnoSentence sent, Embeddings embeddings, int paramOffset, boolean fineTuning,
            VarTensor toUpdate) {
        super(s);
        this.modIn = modIn;
        this.vars = vars;
        this.sent = sent;
        this.embeddings = embeddings;
        this.fineTuning = fineTuning;
        this.feats = feats;
        
        // Model dimensions.
        this.numLabels = vars.calcNumConfigs();
        this.numWordTypes = embeddings.getEmbeddings().getDim(0);
        this.embedDim = embeddings.getEmbeddings().getDim(1);
        this.numFeats = featAlphabet.size(); // K
        
        // Model parameter offset.
        this.embedOffset = paramOffset;
        this.tparamOffset = numWordTypes * embedDim + embedOffset;
        
        this.toUpdate = toUpdate;
    }

    /** Gets the number of model parameters required by this FCM. */
    public int getNumParams() { return numLabels * embedDim * numFeats + numWordTypes * embedDim; }
    
    /**
     * Forward pass:
     * <pre>
     * s_y = \sum_{i=1}^N \sum_{d=1}^D \sum_{k=1}^K T_{y,k,d} f_{i,k} e_{w_i,d} \forall y
     * \psi_{FCM}(y) = exp(s_y)
     * </pre>
     */
    @Override
    public VarTensor forward() {
        assert modIn.getAlgebra().equals(RealAlgebra.getInstance());

        IntDoubleVector modelParams = modIn.getOutput().getModel().getParams();
        // The embeddings must always come first because of how we initialize.
        VTensor embed = new VTensor(modIn.getAlgebra(), embedOffset, modelParams, numWordTypes, embedDim); // e_{w_i,d}
        VTensor tparam = new VTensor(modIn.getAlgebra(), tparamOffset, modelParams, numLabels, embedDim, numFeats); // T_{y,k,d}

        scores = new VarTensor(RealAlgebra.getInstance(), vars);
        assert scores.size() == numLabels;

        // Loop over tokens.
        for (int i=0; i<sent.size(); i++) {
            int w_i = sent.getEmbedId(i);
            if (w_i == -1) { continue; }
            FeatureVector f_i = feats.get(i);
            if (f_i.getUsed() == 0) { continue; }
            // Loop over labels.
            for (int y=0; y<numLabels; y++) {
                // Loop over embedding dimensions.
                for (int d=0; d<embedDim; d++) {
                    double e_wi_d = embed.get(w_i, d);
                    // Loop over (sparse) features.
                    for(int j=0; j<f_i.getUsed(); j++) {
                        int k = f_i.getInternalIndices()[j];
                        double f_ik = f_i.getInternalValues()[j];
                        assert 0 <= k && k < numFeats;
                        // Add to the factor score.
                        // s_y += T_{y,k,d} f_{i,k} e_{w_i,d} \forall y
                        double score = tparam.get(y, d, k) * e_wi_d * f_ik;
                        scores.add(score, y);
                        
                        // Commented for speed:
                        // if (log.isTraceEnabled()) {
                        //     log.trace("i={} y={} d={} k={} s_y={} T_{y,k,d}={} e_{w_i,d}={} f_{i,k}={}", 
                        //             i, y, d, k, score, tparam.get(y, d, k), e_wi_d, f_ik);
                        // }
                    }
                }
            }
        }
        
        // TODO: This special case code should move to Exp.java -- i.e. support conversion of the Algebra there.
        //
        // Convert scores, s_y, to factor values, exp(s_y).
        // \psi_{FCM}(y) = exp(s_y)
        VarTensor fac = new VarTensor(s, scores.getVars());
        for (int y=0; y<scores.size(); y++) {
            fac.setValue(y, s.fromLogProb(scores.getValue(y)));
        }
        
        if (toUpdate != null) {
            // HACK: to update the containing FcmFactor for this FcmModule.
            for (int c=0; c<fac.size(); c++) {
                toUpdate.setValue(c, scores.getValue(c)); //fac.getAlgebra().toLogProb(fac.getValue(c)));
            }
        }
        return y = fac;
    }

    /**
     * Backward pass:
     * <pre>
     * dG/s_y = dG/d\psi_{FCM}(y) exp(s_y)
     * dG/dT_{y,k,d} = dG/ds_y ds_y/dT_{y,k,d} 
     *               = dG/ds_y (\sum_{i=1}^N f_{i,k} e_{w_i,d}) \forall y,k,d
     * dG/de_{w_i,d} = \sum_y dG/ds_y ds_y/dT_{y,k,d} 
     *               = \sum_y dG/ds_y (\sum_{k=1}^K T_{y,k,d} f_{i,k}) \forall i,d
     * </pre>
     */
    @Override
    public void backward() {
        IntDoubleVector modelParams = modIn.getOutput().getModel().getParams();
        VTensor embed = new VTensor(modIn.getAlgebra(), embedOffset, modelParams, numWordTypes, embedDim); // e_{w_i,d}
        VTensor tparam = new VTensor(modIn.getAlgebra(), tparamOffset, modelParams, numLabels, embedDim, numFeats); // T_{y,k,d}
        IntDoubleVector modelParamsAdj = modIn.getOutputAdj().getModel().getParams();
        VTensor embedAdj = fineTuning ? new VTensor(modIn.getAlgebra(), embedOffset, modelParamsAdj, numWordTypes, embedDim) : null; // e_{w_i,d}
        VTensor tparamAdj = new VTensor(modIn.getAlgebra(), tparamOffset, modelParamsAdj, numLabels, embedDim, numFeats); // T_{y,k,d}
        
        // Backprop to scores.
        // dG/s_y = dG/d\psi_{FCM}(y) exp(s_y)
        VarTensor facAdj = getOutputAdj();
        assert facAdj.size() == numLabels;
        VarTensor scoresAdj = new VarTensor(RealAlgebra.getInstance(), facAdj.getVars());
        for (int y=0; y<scoresAdj.size(); y++) {
            // Compute in the output semiring.
            double adj_psi_y = facAdj.getValue(y);
            double adj_s_y = s.times(adj_psi_y, s.fromLogProb(scores.getValue(y)));
            scoresAdj.setValue(y, s.toReal(adj_s_y));
        }
        
        // Backprop to tensor parameters, T, and (optionally) embedding parameters, e.

        // Loop over tokens.
        for (int i=0; i<sent.size(); i++) {
            int w_i = sent.getEmbedId(i);
            if (w_i == -1) { continue; }
            FeatureVector f_i = feats.get(i);
            if (f_i.getUsed() == 0) { continue; }
            // Loop over labels.
            for (int y=0; y<numLabels; y++) {
                // Loop over embedding dimensions.
                for (int d=0; d<embedDim; d++) {
                    double e_wi_d = embed.get(w_i, d);
                    // Loop over (sparse) features.
                    for(int j=0; j<f_i.getUsed(); j++) {
                        int k = f_i.getInternalIndices()[j];
                        double f_ik = f_i.getInternalValues()[j];
                        assert 0 <= k && k < numFeats;

                        // Backprop
                        // dG/dT_{y,k,d} = dG/ds_y ds_y/dT_{y,k,d} 
                        //               = dG/ds_y (\sum_{i=1}^N f_{i,k} e_{w_i,d}) \forall y,k,d
                        double tadd = scoresAdj.get(y) * f_ik * e_wi_d;
                        tparamAdj.add(tadd, y, d, k);
                        if (fineTuning) {
                            // dG/de_{w_i,d} = \sum_y dG/ds_y ds_y/dT_{y,k,d} 
                            //               = \sum_y dG/ds_y (\sum_{k=1}^K T_{y,k,d} f_{i,k}) \forall i,d
                            double eadd = scoresAdj.get(y) * tparam.get(y, d, k) * f_ik;
                            embedAdj.add(eadd, w_i, d);
                        }
                        if (log.isTraceEnabled()) {
                            log.trace("tadd={} dG/ds_y={} T_{y,k,d}={} e_{w_i,d}={} f_{i,k}={}", 
                                    tadd, scoresAdj.get(y), tparam.get(y, d, k), e_wi_d, f_ik);
                        }
                    }
                }
            }
        }
    }

    @Override
    public List<? extends Module<? extends MVec>> getInputs() {
        return QLists.getList(modIn);
    }

    /** Initializes a model with embeddings. */
    public static void initModelWithEmbeds(Embeddings embeddings, FgModel model, ObsFeatureConjoiner ofc) {
        if (ofc.getReservedMax() == 0) {
            // There are no embeddings as model parameters.
            return;
        }
        int offset = ofc.getReservedOffset();
        Tensor e = embeddings.getEmbeddings();
        for (int i=0; i<e.size(); i++) {
            if (offset + i >= model.getNumParams()) { 
                throw new RuntimeException("Invalid model parameter index: " + (offset + i));
            }
            model.getParams().set(offset + i, e.getValue(i));
        }
        log.debug("Initialized model parameters in range [{}, {}]", offset, offset+e.size());
    }
    
}
