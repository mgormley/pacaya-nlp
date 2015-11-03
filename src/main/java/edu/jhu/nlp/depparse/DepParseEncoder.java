package edu.jhu.nlp.depparse;

import edu.jhu.nlp.CorpusStatistics;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.data.simple.IntAnnoSentence;
import edu.jhu.nlp.depparse.BitshiftDepParseFeatureExtractor.BitshiftDepParseFeatureExtractorPrm;
import edu.jhu.nlp.depparse.DepParseFactorGraphBuilder.DepParseFactorGraphBuilderPrm;
import edu.jhu.nlp.depparse.DepParseFeatureExtractor.DepParseFeatureExtractorPrm;
import edu.jhu.pacaya.gm.app.Encoder;
import edu.jhu.pacaya.gm.data.LFgExample;
import edu.jhu.pacaya.gm.data.LabeledFgExample;
import edu.jhu.pacaya.gm.data.UFgExample;
import edu.jhu.pacaya.gm.data.UnlabeledFgExample;
import edu.jhu.pacaya.gm.feat.ObsFeatureConjoiner;
import edu.jhu.pacaya.gm.model.FactorGraph;
import edu.jhu.pacaya.gm.model.Var.VarType;
import edu.jhu.pacaya.gm.model.VarConfig;
import edu.jhu.pacaya.gm.model.globalfac.LinkVar;

/**
 * Encodes a dependency tree factor graph and variable assignment from the words and pruning mask
 * from an {@link AnnoSentence} and the gold parents array.
 * 
 * @author mgormley
 */
public class DepParseEncoder implements Encoder<IntAnnoSentence, int[]> {

    public static class DepParseEncoderPrm {
        // TODO: Fill w/non-null values.
        public DepParseFeatureExtractorPrm dpFePrm = null;
        public BitshiftDepParseFeatureExtractorPrm bsDpFePrm = null;
        public DepParseFactorGraphBuilderPrm dpPrm = null;
    }
    
    private DepParseEncoderPrm prm;
    private CorpusStatistics cs;
    private ObsFeatureConjoiner ofc;
    
    public DepParseEncoder(DepParseEncoderPrm prm, CorpusStatistics cs, ObsFeatureConjoiner ofc) {
        this.cs = cs;
        this.ofc = ofc;
        this.prm = prm;
    }

    @Override
    public LFgExample encode(IntAnnoSentence isent, int[] parents) {
        return getExample(isent, parents, true);
    }

    @Override
    public UFgExample encode(IntAnnoSentence isent) {
        return getExample(isent, null, false);
    }

    private LFgExample getExample(IntAnnoSentence isent, int[] parents, boolean labeledExample) {
        FactorGraph fg = new FactorGraph();
        DepParseFactorGraphBuilder dp = new DepParseFactorGraphBuilder(prm.dpPrm);
        dp.build(isent, fg, cs, ofc);
        
        VarConfig goldConfig = new VarConfig();
        DepParseEncoder.addDepParseTrainAssignment(parents, dp, goldConfig);
        if (labeledExample) {
            return new LabeledFgExample(fg, goldConfig);
        } else {
            return new UnlabeledFgExample(fg);
        }
    }

    /** Add all the training data assignments to the link variables, if they are not latent. */
    public static void addDepParseTrainAssignment(int[] parents, DepParseFactorGraphBuilder dp, VarConfig vc) {
        int n = parents.length;
        // We include the case where the parent is the Wall node (position -1).
        for (int p=-1; p<n; p++) {
            for (int c=0; c<n; c++) {
                if (c != p && dp.getLinkVar(p, c) != null) {
                    LinkVar linkVar = dp.getLinkVar(p, c);
                    if (linkVar.getType() != VarType.LATENT) {
                        // Syntactic head, from dependency parse.
                        int state;
                        if (parents[c] != p) {
                            state = LinkVar.FALSE;
                        } else {
                            state = LinkVar.TRUE;
                        }
                        vc.put(linkVar, state);
                    }
                }
            }
        }
    }

}
