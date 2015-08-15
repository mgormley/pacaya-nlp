package edu.jhu.nlp.relations;

import java.util.List;

import edu.jhu.nlp.CorpusStatistics;
import edu.jhu.nlp.data.NerMention;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.relations.RelationsFactorGraphBuilder.RelVar;
import edu.jhu.nlp.relations.RelationsFactorGraphBuilder.RelationsFactorGraphBuilderPrm;
import edu.jhu.pacaya.gm.app.Encoder;
import edu.jhu.pacaya.gm.data.LFgExample;
import edu.jhu.pacaya.gm.data.LabeledFgExample;
import edu.jhu.pacaya.gm.data.UFgExample;
import edu.jhu.pacaya.gm.data.UnlabeledFgExample;
import edu.jhu.pacaya.gm.feat.ObsFeatureCache;
import edu.jhu.pacaya.gm.feat.ObsFeatureConjoiner;
import edu.jhu.pacaya.gm.feat.ObsFeatureExtractor;
import edu.jhu.pacaya.gm.model.FactorGraph;
import edu.jhu.pacaya.gm.model.VarConfig;
import edu.jhu.prim.tuple.Pair;

public class RelationsEncoder implements Encoder<AnnoSentence, List<String>> {
    
    private RelationsFactorGraphBuilderPrm prm;
    private CorpusStatistics cs;
    private ObsFeatureConjoiner ofc;
    
    public RelationsEncoder(RelationsFactorGraphBuilderPrm prm, CorpusStatistics cs, ObsFeatureConjoiner ofc) {
        this.prm = prm;
        this.cs = cs;
        this.ofc = ofc;
    }
    
    @Override
    public LFgExample encode(AnnoSentence sent, List<String> rels) {
        return getExample(sent, rels, true);
    }

    @Override
    public UFgExample encode(AnnoSentence sent) {
        return getExample(sent, null, false);
    }

    private LFgExample getExample(AnnoSentence sent, List<String> rels, boolean labeledExample) {
        RelationsFactorGraphBuilder rfgb = new RelationsFactorGraphBuilder(prm);
        FactorGraph fg = new FactorGraph();
        rfgb.build(sent, ofc, fg, cs);
        
        VarConfig vc = new VarConfig();
        if (rels != null) {
            addRelVarAssignments(sent, rels, rfgb, vc);
        }
        
        if (labeledExample) {
            return new LabeledFgExample(fg, vc);
        } else {
            return new UnlabeledFgExample(fg);
        }
    }

    public static void addRelVarAssignments(AnnoSentence sent, List<String> rels, RelationsFactorGraphBuilder rfgb,
            VarConfig vc) {
        List<Pair<NerMention, NerMention>> nePairs = sent.getNePairs();
        for (RelVar var : rfgb.getRelVars()) {    	
    		NerMention ne1 = var.ment1;
    		NerMention ne2 = var.ment2;
            int k = nePairs.indexOf(new Pair<NerMention,NerMention>(ne1, ne2));
            String relation = rels.get(k);
            assert var != null;
            vc.put(var, relation);
    	}
    }
        
}
