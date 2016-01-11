package edu.jhu.nlp.srl;

import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.CorpusStatistics;
import edu.jhu.nlp.ObsFeTypedFactor;
import edu.jhu.nlp.data.simple.IntAnnoSentence;
import edu.jhu.nlp.depparse.DepParseFactorGraphBuilder.DepParseFactorTemplate;
import edu.jhu.nlp.features.FeaturizedSentence;
import edu.jhu.nlp.features.IntTemplateFeatureExtractor;
import edu.jhu.nlp.features.LocalObservations;
import edu.jhu.nlp.features.TemplateFeatureExtractor;
import edu.jhu.nlp.features.TemplateLanguage.FeatTemplate;
import edu.jhu.nlp.features.TemplateSets;
import edu.jhu.nlp.joint.JointNlpFactorGraph.JointFactorTemplate;
import edu.jhu.nlp.relations.FeatureUtils;
import edu.jhu.nlp.sprl.SprlFactorGraphBuilder.SprlFactorType;
import edu.jhu.nlp.sprl.SprlFactorGraphBuilder.SprlVar;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.RoleVar;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.SenseVar;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.SrlFactorTemplate;
import edu.jhu.pacaya.gm.feat.FeatureVector;
import edu.jhu.pacaya.gm.feat.ObsFeExpFamFactor;
import edu.jhu.pacaya.gm.feat.ObsFeatureConjoiner;
import edu.jhu.pacaya.gm.feat.ObsFeatureExtractor;
import edu.jhu.pacaya.gm.model.Var;
import edu.jhu.pacaya.gm.model.VarSet;
import edu.jhu.pacaya.gm.model.globalfac.LinkVar;
import edu.jhu.pacaya.util.FeatureNames;
import edu.jhu.pacaya.util.Prm;
import edu.jhu.prim.list.IntArrayList;

/**
 * Feature extractor for SRL. All the "real" feature extraction is done in
 * SentFeatureExtraction which considers only the observations.
 * 
 * @author mgormley
 * @author mmitchell
 */
public class SrlFeatureExtractor implements ObsFeatureExtractor {

    public static class SrlFeatureExtractorPrm extends Prm {
        private static final long serialVersionUID = 1L;
        /**
         * For testing only: this will ensure that the only feature returned is
         * the bias feature.
         */
        public boolean biasOnly = false;
        /** Whether to use ONLY feature templates. */
        public boolean useTemplates = true;
        /** Feature templates. */
        public List<FeatTemplate> senseTemplates = TemplateSets.getNaradowskySenseUnigramFeatureTemplates();
        public List<FeatTemplate> argTemplates = TemplateSets.getNaradowskyArgUnigramFeatureTemplates();
        /**
         * The value of the mod for use in the feature hashing trick. If <= 0,
         * feature-hashing will be disabled.
         */
        public int featureHashMod = -1;
    }

    private static final Logger log = LoggerFactory.getLogger(SrlFeatureExtractor.class);
    private static final int BIAS_HASH = "BIAS_FEATURE".hashCode();

    private SrlFeatureExtractorPrm prm;
    private TemplateFeatureExtractor ext;
    private IntTemplateFeatureExtractor intExt;
    private ObsFeatureConjoiner ofc;

    public SrlFeatureExtractor(SrlFeatureExtractorPrm prm, IntAnnoSentence isent, CorpusStatistics cs,
            ObsFeatureConjoiner ofc) {
        this.prm = prm;
        if (!prm.biasOnly) {
            if (prm.featureHashMod <= 0) {
                FeaturizedSentence fSent = new FeaturizedSentence(isent.getAnnoSentence(), cs);
                ext = new TemplateFeatureExtractor(fSent, cs);
            } else {
                intExt = new IntTemplateFeatureExtractor(isent, cs);
            }
        }
        this.ofc = ofc;
    }

    @Override
    public FeatureVector calcObsFeatureVector(ObsFeExpFamFactor factor) {
        ObsFeTypedFactor f = (ObsFeTypedFactor) factor;
        Enum<?> ft = f.getFactorType();
        VarSet vars = f.getVars();

        // Get the observation features.
        int parent = Integer.MIN_VALUE;
        int child = Integer.MIN_VALUE;
        List<FeatTemplate> tpls;
        if (ft == JointFactorTemplate.LINK_ROLE_BINARY || ft == DepParseFactorTemplate.UNARY
                || ft == SrlFactorTemplate.ROLE_UNARY || ft == SrlFactorTemplate.SENSE_ROLE_BINARY
                || ft == JointFactorTemplate.ROLE_C_TAG_BINARY || ft == JointFactorTemplate.ROLE_P_TAG_BINARY
                || ft == JointFactorTemplate.ROLE_SPRL_BINARY || ft == SprlFactorType.SPRL_UNARY
                || ft == SprlFactorType.SPRL_PAIRWISE) {
            tpls = prm.argTemplates;
            // Look at the variables to determine the parent and child.
            for (int i = 0; i < vars.size(); i++) {
                Var var = vars.get(i);
                if (var instanceof LinkVar) {
                    parent = ((LinkVar) var).getParent();
                    child = ((LinkVar) var).getChild();
                    break;
                } else if (var instanceof RoleVar) {
                    parent = ((RoleVar) var).getParent();
                    child = ((RoleVar) var).getChild();
                    break;
                } else if (var instanceof SprlVar) {
                    parent = ((SprlVar) var).getPred();
                    child = ((SprlVar) var).getArg();
                    break;
                }
            }
            if (parent == Integer.MIN_VALUE && child == Integer.MIN_VALUE) {
                throw new RuntimeException("Unknown variable type for ft: " + ft);
            }
        } else if (ft == SrlFactorTemplate.SENSE_UNARY) {
            tpls = prm.senseTemplates;
            SenseVar var = (SenseVar) vars.iterator().next();
            parent = var.getParent();
        } else {
            throw new RuntimeException("Unsupported template: " + ft);
        }

        FeatureNames alphabet = ofc.getTemplates().getTemplate(f).getAlphabet();
        return getSrlFeats(alphabet, parent, child, tpls);
    }

    /**
     * Gets SRL features.
     * 
     * @param alphabet
     *            (OPTIONAL) The int-to-string mapping used only if extracting
     *            string features.
     * @param parent
     *            The index of the parent.
     * @param child
     *            (OPTIONAL) The index of the child.
     * @param tpls
     *            The feature template list.
     * @return The extracted features.
     */
    public FeatureVector getSrlFeats(FeatureNames alphabet, int parent, int child, List<FeatTemplate> tpls) {
        if (prm.biasOnly || ext != null) {
            ArrayList<String> obsFeats = new ArrayList<>();
            // Get features on the observations for a pair of words.
            // IMPORTANT NOTE: We include the case where the parent is the Wall
            // node (position -1).
            if (!prm.biasOnly) {
                ext.addFeatures(tpls, LocalObservations.newPidxCidx(parent, child), obsFeats);
            }

            if (log.isTraceEnabled()) {
                log.trace("Num obs features in factor: " + obsFeats.size());
            }

            // The bias features are used to ensure that at least one feature
            // fires for each variable configuration.
            ArrayList<String> biasFeats = new ArrayList<String>();
            biasFeats.add("BIAS_FEATURE");

            // Add the bias features.
            FeatureVector fv = new FeatureVector(biasFeats.size() + obsFeats.size());
            FeatureUtils.addFeatures(biasFeats, alphabet, fv, true, prm.featureHashMod);

            // Add the other features.
            FeatureUtils.addFeatures(obsFeats, alphabet, fv, false, prm.featureHashMod);
            return fv;
        } else {
            IntArrayList obsFeats = new IntArrayList();
            if (!prm.biasOnly) {
                intExt.addFeatures(tpls, LocalObservations.newPidxCidx(parent, child), obsFeats);
            }
            obsFeats.add(BIAS_HASH);
            FeatureVector fv = new FeatureVector(obsFeats.size());
            FeatureUtils.addFeatures(obsFeats, fv, prm.featureHashMod, alphabet);
            return fv;
        }
    }

}
