package edu.jhu.nlp.depparse;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang3.ArrayUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.CorpusStatistics;
import edu.jhu.nlp.FeTypedFactor;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.depparse.DepParseFactorGraphBuilder.DepParseFactorTemplate;
import edu.jhu.nlp.depparse.DepParseFactorGraphBuilder.GraFeTypedFactor;
import edu.jhu.nlp.depparse.DepParseFactorGraphBuilder.SibFeTypedFactor;
import edu.jhu.nlp.features.FeaturizedSentence;
import edu.jhu.nlp.features.LocalObservations;
import edu.jhu.nlp.features.TemplateFeatureExtractor;
import edu.jhu.nlp.features.TemplateLanguage.FeatTemplate;
import edu.jhu.nlp.features.TemplateSets;
import edu.jhu.nlp.relations.FeatureUtils;
import edu.jhu.pacaya.gm.data.UFgExample;
import edu.jhu.pacaya.gm.feat.FeatureExtractor;
import edu.jhu.pacaya.gm.feat.FeatureVector;
import edu.jhu.pacaya.gm.model.Factor;
import edu.jhu.pacaya.gm.model.FeExpFamFactor;
import edu.jhu.pacaya.gm.model.Var;
import edu.jhu.pacaya.gm.model.VarConfig;
import edu.jhu.pacaya.gm.model.VarSet;
import edu.jhu.pacaya.gm.model.Var.VarType;
import edu.jhu.pacaya.gm.model.globalfac.LinkVar;
import edu.jhu.pacaya.util.FeatureNames;
import edu.jhu.pacaya.util.Prm;

public class DepParseFeatureExtractor implements FeatureExtractor {

    public static class DepParseFeatureExtractorPrm extends Prm {
        private static final long serialVersionUID = 1L;
        /** Feature options. */
        public List<FeatTemplate> firstOrderTpls = TemplateSets.getNaradowskyArgUnigramFeatureTemplates();
        public List<FeatTemplate> secondOrderTpls = TemplateSets.getFromResource(TemplateSets.carreras07Dep2FeatsResource);
        public boolean biasOnly = false;
        /** The value of the mod for use in the feature hashing trick. If <= 0, feature-hashing will be disabled. */
        public int featureHashMod = -1;
        /** Whether to create human interpretable feature names when possible. */
        public boolean humanReadable = true;
        /** Whether to only include non-bias features on edges in the tree. */
        public boolean onlyTrueEdges = true;
        /** Whether to only include the bias feature on edges in the tree. */ 
        public boolean onlyTrueBias = true;
        /** Whether to use only the fast-to-extract feature set. */
        public boolean onlyFast = false;
    }
    
    private static final Logger log = LoggerFactory.getLogger(DepParseFeatureExtractor.class); 
    
    private DepParseFeatureExtractorPrm prm;
    private FeatureNames alphabet;
    private TemplateFeatureExtractor ext;
    
    public DepParseFeatureExtractor(DepParseFeatureExtractorPrm prm, AnnoSentence sent, CorpusStatistics cs, FeatureNames alphabet) {
        this.prm = prm;
        FeaturizedSentence fSent = new FeaturizedSentence(sent, cs);
        ext = new TemplateFeatureExtractor(fSent, cs);
        this.alphabet = alphabet;
    }

    @Override
    public void init(UFgExample ex) { }
    
    private final FeatureVector emptyFv = new FeatureVector();

    @Override
    public FeatureVector calcFeatureVector(FeExpFamFactor factor, int configId) {
        FeTypedFactor f = (FeTypedFactor) factor;
        Enum<?> ft = f.getFactorType();
        VarSet vars = f.getVars();
        
        int[] vc = vars.getVarConfigAsArray(configId);
        if (prm.onlyTrueBias && prm.onlyTrueEdges && ArrayUtils.contains(vc, LinkVar.FALSE)) {
            return emptyFv;
        }

        ArrayList<String> obsFeats = new ArrayList<String>();
        if (!prm.onlyTrueEdges || !ArrayUtils.contains(vc, LinkVar.FALSE)) {                
            // Get the observation features.
            if (ft == DepParseFactorTemplate.UNARY) {
                // Look at the variables to determine the parent and child.
                LinkVar var = (LinkVar) vars.get(0);
                int pidx = var.getParent();
                int cidx = var.getChild();
                ext.addFeatures(prm.firstOrderTpls, LocalObservations.newPidxCidx(pidx, cidx), obsFeats);
            } else if (ft == DepParseFactorTemplate.ARBITRARY_SIBLING) {
                SibFeTypedFactor f2 = (SibFeTypedFactor)f;
                ext.addFeatures(prm.secondOrderTpls, LocalObservations.newPidxCidxMidx(f2.p, f2.c, f2.s), obsFeats);
            } else if (ft == DepParseFactorTemplate.GRANDPARENT) {
                GraFeTypedFactor f2 = (GraFeTypedFactor)f;
                ext.addFeatures(prm.secondOrderTpls, LocalObservations.newPidxCidxMidx(f2.p, f2.c, f2.g), obsFeats);
            } else {
                throw new RuntimeException("Unsupported template: " + ft);
            }
        }
        
        // Create prefix containing the states of the variables.
        String prefix = ft + "_" + configId + "_";
        
        // Add the bias features.
        // The bias features are used to ensure that at least one feature fires for each variable configuration.
        ArrayList<String> biasFeats = new ArrayList<String>();
        biasFeats.add("BIAS_FEATURE");
        biasFeats.add(prefix + "BIAS_FEATURE");

        // Add the bias features.
        FeatureVector fv = new FeatureVector(biasFeats.size() + obsFeats.size());
        FeatureUtils.addFeatures(biasFeats, alphabet, fv, true, prm.featureHashMod);
        
        // Add the other features.
        FeatureUtils.addPrefix(obsFeats, prefix);
        FeatureUtils.addFeatures(obsFeats, alphabet, fv, false, prm.featureHashMod);
        
        return fv;
    }
    
}
