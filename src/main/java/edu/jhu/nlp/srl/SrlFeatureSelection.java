package edu.jhu.nlp.srl;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.Annotator;
import edu.jhu.nlp.CorpusStatistics.CorpusStatisticsPrm;
import edu.jhu.nlp.Trainable;
import edu.jhu.nlp.data.simple.AnnoSentenceCollection;
import edu.jhu.nlp.data.simple.CorpusHandler;
import edu.jhu.nlp.depparse.DepParseFeatureExtractor.DepParseFeatureExtractorPrm;
import edu.jhu.nlp.features.TemplateLanguage;
import edu.jhu.nlp.features.TemplateLanguage.AT;
import edu.jhu.nlp.features.TemplateLanguage.FeatTemplate;
import edu.jhu.nlp.features.TemplateLanguage.FeatTemplate0;
import edu.jhu.nlp.features.TemplateLanguage.JoinTemplate;
import edu.jhu.nlp.features.TemplateLanguage.OtherFeat;
import edu.jhu.nlp.features.TemplateWriter;
import edu.jhu.nlp.joint.IGFeatureTemplateSelector;
import edu.jhu.nlp.joint.IGFeatureTemplateSelector.IGFeatureTemplateSelectorPrm;
import edu.jhu.nlp.joint.IGFeatureTemplateSelector.SrlFeatTemplates;
import edu.jhu.nlp.joint.JointNlpFactorGraph.JointNlpFactorGraphPrm;
import edu.jhu.nlp.joint.JointNlpRunner;
import edu.jhu.nlp.srl.SrlFeatureExtractor.SrlFeatureExtractorPrm;

/**
 * Train-time only "annotator" for feature selection. This modifies the feature templates on the
 * given {@link JointNlpFeatureExtractorPrm} in place.
 * 
 * TODO: Deprecate this class as it has static dependencies on JointNlpRunner.
 * 
 * @author mgormley
 */
public class SrlFeatureSelection implements Annotator, Trainable {

    private static final long serialVersionUID = 1L;
    private static final Logger log = LoggerFactory.getLogger(SrlFeatureSelection.class);
    private transient JointNlpFactorGraphPrm fgPrm;

    public SrlFeatureSelection(JointNlpFactorGraphPrm fgPrm) {
        this.fgPrm = fgPrm;
    }

    @Override
    public void annotate(AnnoSentenceCollection sents) {
        // Do nothing. This only runs at training time.
    }

    @Override
    public void train(AnnoSentenceCollection trainInput, AnnoSentenceCollection trainGold,
            AnnoSentenceCollection devInput, AnnoSentenceCollection devGold) {
        featureSelection(trainInput, trainGold, fgPrm);
    }

    /**
     * Do feature selection and update fePrm with the chosen feature templates.
     */
    // TODO: This method does far more than feature selection and should be simplified.
    private static void featureSelection(AnnoSentenceCollection inputSents, AnnoSentenceCollection goldSents, JointNlpFactorGraphPrm fgPrm)  {
        SrlFeatureExtractorPrm srlFePrm = fgPrm.srlPrm.srlFePrm;
        // Remove annotation types from the features which are explicitly excluded.
        removeAts(fgPrm);
        if (JointNlpRunner.useTemplates && JointNlpRunner.featureSelection) {
            CorpusStatisticsPrm csPrm = JointNlpRunner.getCorpusStatisticsPrm();
            
            IGFeatureTemplateSelectorPrm prm = JointNlpRunner.getInformationGainFeatureSelectorPrm();
            SrlFeatTemplates sft = new SrlFeatTemplates(srlFePrm.senseTemplates, srlFePrm.argTemplates, null);
            IGFeatureTemplateSelector ig = new IGFeatureTemplateSelector(prm);
            sft = ig.getFeatTemplatesForSrl(inputSents, goldSents, csPrm, sft);
            // Set the chosen templates for SRL.
            fgPrm.srlPrm.srlFePrm.senseTemplates = sft.srlSense;
            fgPrm.srlPrm.srlFePrm.argTemplates = sft.srlArg;
        }
        if (JointNlpRunner.srlExtraArgFeats) {
            // For the original ACL'14 experiments, we allowed factors between "PREDICTED" and
            // "OBSERVED" variables. Here, we add the equivalent features for such a factor explicitly.
            List<FeatTemplate> argFeats = fgPrm.srlPrm.srlFePrm.argTemplates;
            List<FeatTemplate> newFeats = new ArrayList<>();
            for (FeatTemplate tpl : argFeats) {
                newFeats.add(new JoinTemplate(tpl, new FeatTemplate0(OtherFeat.DIR_EDGE)));            
            }
            argFeats.addAll(newFeats);
        }
        if (CorpusHandler.getPredLatAts().contains(AT.SRL) && JointNlpRunner.acl14DepFeats) {
            // Set the chosen templates for dependencing parsing.
            fgPrm.dpPrm.dpFePrm.firstOrderTpls = srlFePrm.argTemplates;
        }
        if (JointNlpRunner.useTemplates) {
            log.info("Num sense feature templates: " + srlFePrm.senseTemplates.size());
            log.info("Num arg feature templates: " + srlFePrm.argTemplates.size());
            if (JointNlpRunner.senseFeatTplsOut != null) {
                TemplateWriter.write(JointNlpRunner.senseFeatTplsOut, srlFePrm.senseTemplates);
            }
            if (JointNlpRunner.argFeatTplsOut != null) {
                TemplateWriter.write(JointNlpRunner.argFeatTplsOut, srlFePrm.argTemplates);
            }
        }
    }

    private static void removeAts(JointNlpFactorGraphPrm fgPrm) {
        Set<AT> ats = new HashSet<>();
        ats.addAll(CorpusHandler.getRemoveAts());
        ats.addAll(CorpusHandler.getLatAts()); 
        ats.addAll(CorpusHandler.getPredAts());
        if (JointNlpRunner.brownClusters == null) {
            // Filter out the Brown cluster features.
            log.warn("Filtering out Brown cluster features.");
            ats.add(AT.BROWN);
        }
        for (AT at : ats) {
            SrlFeatureExtractorPrm srlFePrm = fgPrm.srlPrm.srlFePrm;
            DepParseFeatureExtractorPrm dpFePrm = fgPrm.dpPrm.dpFePrm;
            fgPrm.srlPrm.srlFePrm.senseTemplates = TemplateLanguage.filterOutRequiring(srlFePrm.senseTemplates, at);
            fgPrm.srlPrm.srlFePrm.argTemplates   = TemplateLanguage.filterOutRequiring(srlFePrm.argTemplates, at);
            fgPrm.dpPrm.dpFePrm.firstOrderTpls = TemplateLanguage.filterOutRequiring(dpFePrm.firstOrderTpls, at);
            fgPrm.dpPrm.dpFePrm.secondOrderTpls   = TemplateLanguage.filterOutRequiring(dpFePrm.secondOrderTpls, at);
            fgPrm.posPrm.templates = TemplateLanguage.filterOutRequiring(fgPrm.posPrm.templates, at);
        }
    }
    
    @Override
    public Set<AT> getAnnoTypes() {
        return Collections.emptySet();
    }
    
}