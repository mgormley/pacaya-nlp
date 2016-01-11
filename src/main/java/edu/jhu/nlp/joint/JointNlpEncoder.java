package edu.jhu.nlp.joint;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.CorpusStatistics;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.data.simple.AnnoSentenceCollection;
import edu.jhu.nlp.depparse.DepParseEncoder;
import edu.jhu.nlp.features.TemplateLanguage;
import edu.jhu.nlp.joint.JointNlpFactorGraph.IsArgLabel;
import edu.jhu.nlp.joint.JointNlpFactorGraph.JointNlpFactorGraphPrm;
import edu.jhu.pacaya.gm.app.Encoder;
import edu.jhu.pacaya.gm.data.LFgExample;
import edu.jhu.pacaya.gm.data.LabeledFgExample;
import edu.jhu.pacaya.gm.data.UFgExample;
import edu.jhu.pacaya.gm.data.UnlabeledFgExample;
import edu.jhu.pacaya.gm.feat.FactorTemplateList;
import edu.jhu.pacaya.gm.feat.ObsFeatureConjoiner;
import edu.jhu.pacaya.gm.model.Var;
import edu.jhu.pacaya.gm.model.Var.VarType;
import edu.jhu.pacaya.gm.model.VarConfig;
import edu.jhu.pacaya.util.Prm;
import edu.jhu.prim.tuple.Pair;

/**
 * Encodes a joint NLP factor graph and its variable assignment.
 * @author mgormley
 */
public class JointNlpEncoder implements Encoder<AnnoSentence, AnnoSentence> {

    private static final Logger log = LoggerFactory.getLogger(JointNlpEncoder.class);

    public static class JointNlpEncoderPrm extends Prm {
        private static final long serialVersionUID = 1L;
        public JointNlpFactorGraphPrm fgPrm = new JointNlpFactorGraphPrm();
    }
    
    private JointNlpEncoderPrm prm;
    private CorpusStatistics cs;
    private ObsFeatureConjoiner ofc;
    
    public JointNlpEncoder(JointNlpEncoderPrm prm, CorpusStatistics cs, ObsFeatureConjoiner ofc) {
        this.prm = prm;
        this.cs = cs;
        this.ofc = ofc;
    }

    @Override
    public LFgExample encode(AnnoSentence input, AnnoSentence gold) {
        return getExample(input, gold, true);
    }

    @Override
    public UFgExample encode(AnnoSentence input) {
        return getExample(input, null, false);
    }

    private LFgExample getExample(AnnoSentence sent, AnnoSentence gold, boolean labeledExample) {        
        // Construct the factor graph.
        JointNlpFactorGraph fg = new JointNlpFactorGraph(prm.fgPrm, sent, cs, ofc);
        log.trace("Number of variables: " + fg.getNumVars() + " Number of factors: " + fg.getNumFactors() + " Number of edges: " + fg.getNumEdges());

        // Get the variable assignments given in the training data.
        VarConfig vc = new VarConfig();
        if (prm.fgPrm.includePos && prm.fgPrm.posPrm.posTagVarType != VarType.LATENT) {
            if (gold != null && gold.getPosTags() != null) {
                fg.getPosTagBuilder().addVarAssignments(gold.getPosTags(), vc);;
            }
        }
        if (prm.fgPrm.includeDp && prm.fgPrm.dpPrm.linkVarType != VarType.LATENT) {
            if (gold != null && gold.getParents() != null) {
                DepParseEncoder.addDepParseTrainAssignment(gold.getParents(), fg.getDpBuilder(), vc);
            }
        }
        if (prm.fgPrm.includeSrl && prm.fgPrm.srlPrm.srlVarType != VarType.LATENT) {
            if (gold != null && gold.getSrlGraph() != null) {
                fg.getSrlBuilder().addVarAssignments(gold.getSrlGraph(), vc);
            }
        }

        if (prm.fgPrm.includeSprl) {
            if (gold != null && gold.getSprl() != null) {
                fg.getSprlBuilder().annoToConfig(gold,  vc);
            }
        }

        if (prm.fgPrm.includeRel && prm.fgPrm.relPrm.relVarType != VarType.LATENT) {
            if (gold != null && gold.getRelLabels() != null) {
                fg.getRelBuilder().addVarAssignments(sent, gold.getRelLabels(), vc);
            }
        }
        
        // Create the example.
        LFgExample ex;
        FactorTemplateList fts = ofc.getTemplates();
        if (labeledExample) {
            ex = new LabeledFgExample(fg, vc, fts);
        } else {
            ex = new UnlabeledFgExample(fg, fts);
        }
        return ex;
    }

    public static void checkForRequiredAnnotations(JointNlpEncoderPrm prm, AnnoSentenceCollection sents) {
        try {
            if (sents.size() == 0) { return; }
            // Check that the first sentence has all the required annotation
            // types for the specified feature templates.
            AnnoSentence sent = sents.get(0);
            if (prm.fgPrm.srlPrm.srlFePrm.useTemplates) {
                if (prm.fgPrm.includeSrl) {
                    TemplateLanguage.assertRequiredAnnotationTypes(sent, prm.fgPrm.srlPrm.srlFePrm.senseTemplates);
                    TemplateLanguage.assertRequiredAnnotationTypes(sent, prm.fgPrm.srlPrm.srlFePrm.argTemplates);
                }
            }
            if (prm.fgPrm.includeDp && !prm.fgPrm.dpPrm.dpFePrm.onlyFast) {
                TemplateLanguage.assertRequiredAnnotationTypes(sent, prm.fgPrm.dpPrm.dpFePrm.firstOrderTpls);
                if (prm.fgPrm.dpPrm.grandparentFactors || prm.fgPrm.dpPrm.arbitrarySiblingFactors) {
                    TemplateLanguage.assertRequiredAnnotationTypes(sent, prm.fgPrm.dpPrm.dpFePrm.secondOrderTpls);
                }
            }
        } catch (IllegalStateException e) {
            log.error(e.getMessage());
            log.trace("", e);
        }
    }
    
}
