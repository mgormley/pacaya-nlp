package edu.jhu.nlp.srl;

import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.data.Span;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.fcm.WordFeatures;
import edu.jhu.nlp.features.FeaturizedSentence;
import edu.jhu.nlp.features.FeaturizedTokenPair;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.RoleVar;
import edu.jhu.pacaya.gm.feat.FeatureVector;
import edu.jhu.pacaya.gm.model.VarSet;
import edu.jhu.pacaya.parse.dep.ParentsArray;
import edu.jhu.pacaya.util.FeatureNames;
import edu.jhu.pacaya.util.Prm;
import edu.jhu.pacaya.util.cli.Opt;
import edu.jhu.prim.tuple.Pair;

/**
 * Per-word features used by the FCM.
 * 
 * @author mgormley
 */
public class SrlWordFeatures implements WordFeatures {

    public enum SrlWordFeatType { HEAD_ONLY, HEAD_TYPE, HEAD_TYPE_LOC, HEAD_TYPE_LOC_ST, HEAD_TYPE_LOC_ST_LT, FULL }

    public static class SrlWordFeaturesPrm extends Prm {
        private static final long serialVersionUID = 1L;
        @Opt(hasArg=true, description="The feature set for embeddings.")
        public SrlWordFeatType embFeatType = SrlWordFeatType.FULL;
    }
    
    private static final Logger log = LoggerFactory.getLogger(SrlWordFeatures.class);

    private SrlWordFeaturesPrm prm;
    private AnnoSentence sent;
    private FeatureNames alphabet;
    private FeaturizedSentence fsent;

    public SrlWordFeatures(SrlWordFeaturesPrm prm, AnnoSentence sent, FeatureNames alphabet) {
        this.prm = prm;
        this.sent = sent;
        this.alphabet = alphabet;
        fsent = new FeaturizedSentence(sent, null);
    }

    @Override
    public FeatureNames getAlphabet() {
        return alphabet;
    }
    
    @Override
    public List<FeatureVector> getFeatures(VarSet vars) {
        if (vars.size() != 1 || !(vars.get(0) instanceof RoleVar)) {
            throw new IllegalArgumentException("Expected one var of type " + RoleVar.class);
        }
        return getFeatures((RoleVar)vars.get(0));
    }
    
    public List<FeatureVector> getFeatures(RoleVar rv) {
        List<FeatureVector> fvs = getListOfEmptyFvs(sent.size());
        addEmbeddingFeatures(rv.getParent(), rv.getChild(), fvs);
        return fvs;
    }

    /**
     * Adds the per-word features for the FCM.
     * 
     * @param local The local observations.
     * @param fv Output feature vector
     */
    private void addEmbeddingFeatures(int head, int modifier, List<FeatureVector> fvs) {     
        int left = Math.min(head, modifier);
        int right = Math.max(head, modifier);
        String hC = prefix(sent.getCluster(head), 4);
        String mC = prefix(sent.getCluster(modifier), 4);
        String hD = prefix(sent.getCluster(head), 6);
        String mD = prefix(sent.getCluster(modifier), 6);
        String hC_mC = hC + mC;

        // In the comments below, we use the following abbreviations.
        //- h : is the head
        //- m : is the modifier
        //- l : is the left
        //- r : is the right
        //- hC : equals h.bc0
        //- hD : equals h.bc1
        
        switch (prm.embFeatType) {
        case FULL:
        case HEAD_TYPE_LOC_ST_LT:
            //- {l} \cross {hC, mC, hC_mC}
            addEmbFeat("l-hC:"+hC,      left, fvs);
            addEmbFeat("l-mC:"+mC,      left, fvs);
            addEmbFeat("l-hCmC:"+hC_mC, left, fvs);
            //- {r} \cross {hC, mC, hC_mC}            
            addEmbFeat("r-hC:"+hC,      right, fvs);
            addEmbFeat("r-mC:"+mC,      right, fvs);
            addEmbFeat("r-hCmC:"+hC_mC, right, fvs);
            
        case HEAD_TYPE_LOC_ST:
            //- {h} \cross {hD, mD}
            //- {m} \cross {hD, mD}
            addEmbFeat("h-hD:"+hD,    head, fvs);
            addEmbFeat("h-mD:"+mD,    head, fvs);
            addEmbFeat("m-hD:"+hD,    modifier, fvs);
            addEmbFeat("m-mD:"+mD,    modifier, fvs);
            
        case HEAD_TYPE_LOC:
            //- {btwn} \cross {hC, mC, hC+mC} : btwn = is the word between l and r
            Span btwn = new Span(left+1, right);
            for (int i=btwn.start(); i<btwn.end(); i++) {
                addEmbFeat("in_between", i, fvs);
                addEmbFeat("in_between-hC:"+hC, i, fvs);
                addEmbFeat("in_between-mC:"+mC, i, fvs);
                // COMMENTED OUT: addEmbFeat("in_between-hCmC:"+hC_mC, i, fvs);
            }
            //- {onpath} \cross {hC, mC, hC_mC} : onpath = is the word on the dependency
            //  path between h and m
            if (sent.getParents() != null) {
                FeaturizedTokenPair ftp = fsent.getFeatTokPair(modifier, head);
                List<Pair<Integer, ParentsArray.Dir>> depPath = ftp.getDependencyPath();
                if (depPath != null) {
                    for (Pair<Integer,ParentsArray.Dir> pair : depPath) {
                        int i = pair.get1();
                        addEmbFeat("on_path", i, fvs);
                        addEmbFeat("on_path-hC:"+hC, i, fvs);
                        addEmbFeat("on_path-mC:"+mC, i, fvs);
                        // COMMENTED OUT: addEmbFeat("on_path-hCmC:"+hC_mC, i, fvs);
                    }
                } else {
                    log.trace("No dependency path between mention heads");
                }
            } else {
                log.trace("No dependency tree for sentence");
            }
            
            //     - -1_h: immediately to the left of the head
            //     - +1_h: immediately to the right of the head
            //     - -2_h: two to the left of the head
            //     - +2_h: two to the right of the head
            addEmbFeat("-1_h", head-1, fvs);
            addEmbFeat("+1_h", head+1, fvs);
            addEmbFeat("-2_h", head-2, fvs);
            addEmbFeat("+2_h", head+2, fvs);
            
            //     - -1_m: immediately to the left of the modifier
            //     - +1_m: immediately to the right of the modifier
            //     - -2_m: two to the left of the modifier
            //     - +2_m: two to the right of the modifier
            addEmbFeat("-1_m", modifier-1, fvs);
            addEmbFeat("+1_m", modifier+1, fvs);
            addEmbFeat("-2_m", modifier-2, fvs);
            addEmbFeat("+2_m", modifier+2, fvs);
                                
        case HEAD_TYPE:
            //- {h} \cross {hC, mC, hC_mC}
            addEmbFeat("h-hC:"+hC,      head, fvs);
            addEmbFeat("h-mC:"+mC,      head, fvs);
            addEmbFeat("h-hCmC:"+hC_mC, head, fvs);
            //- {m} \cross {hC, mC, hC_mC}
            addEmbFeat("m-hC:"+hC,      modifier, fvs);
            addEmbFeat("m-mC:"+mC,      modifier, fvs);
            addEmbFeat("m-hCmC:"+hC_mC, modifier, fvs);
            
        case HEAD_ONLY:
            //- h : is the head
            addEmbFeat("h",        head, fvs);
            //- m : is the modifier
            addEmbFeat("m",        modifier, fvs);
        }
    }
    
    private static String prefix(String s, int max) {
        return s.substring(0, Math.min(s.length(), max));
    }

    private void addEmbFeat(String fname, int i, List<FeatureVector> fvs) {
        if (i < 0 || sent.size() <= i) {
            return;
        }
        FeatureVector fv = fvs.get(i);
        int fidx = alphabet.lookupIndex(fname);
        if (fidx != -1) {
            fv.add(fidx, 1.0);
        }
    }
    
}
