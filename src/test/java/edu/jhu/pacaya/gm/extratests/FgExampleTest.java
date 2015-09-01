package edu.jhu.pacaya.gm.extratests;

import static edu.jhu.nlp.data.simple.AnnoSentenceCollection.getSingleton;
import static org.junit.Assert.assertEquals;

import java.util.ArrayList;
import java.util.List;

import org.junit.Test;

import edu.jhu.nlp.CorpusStatistics;
import edu.jhu.nlp.CorpusStatistics.CorpusStatisticsPrm;
import edu.jhu.nlp.data.conll.CoNLL09Sentence;
import edu.jhu.nlp.data.conll.CoNLL09Token;
import edu.jhu.nlp.data.simple.AnnoSentenceCollection;
import edu.jhu.nlp.joint.JointNlpFgExamplesBuilder;
import edu.jhu.nlp.joint.JointNlpFgExamplesBuilder.JointNlpFgExampleBuilderPrm;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.RoleStructure;
import edu.jhu.pacaya.gm.data.FgExampleList;
import edu.jhu.pacaya.gm.data.LFgExample;
import edu.jhu.pacaya.gm.feat.FactorTemplateList;
import edu.jhu.pacaya.gm.feat.ObsFeatureConjoiner;
import edu.jhu.pacaya.gm.feat.ObsFeatureConjoiner.ObsFeatureConjoinerPrm;
import edu.jhu.pacaya.gm.model.Factor;
import edu.jhu.pacaya.gm.model.FactorGraph;
import edu.jhu.pacaya.gm.model.Var.VarType;
import edu.jhu.pacaya.gm.train.MarginalLogLikelihood;
import edu.jhu.pacaya.util.collections.QLists;

public class FgExampleTest {

    @Test
    public void testClampedFactorGraphs() {
        List<CoNLL09Token> tokens = new ArrayList<CoNLL09Token>();
        //tokens.add(new CoNLL09Token(1, "the", "_", "_", "Det", "_", getList("feat"), getList("feat") , 2, 2, "det", "_", false, "_", new ArrayList<String>()));
        //tokens.add(new CoNLL09Token(id, form, lemma, plemma, pos, ppos, feat, pfeat, head, phead, deprel, pdeprel, fillpred, pred, apreds));
        tokens.add(new CoNLL09Token(1, "the", "_", "_", "Det", "_", QLists.getList("feat"), QLists.getList("feat") , 2, 2, "det", "_", false, "_", QLists.getList("_")));
        tokens.add(new CoNLL09Token(2, "dog", "_", "_", "N", "_", QLists.getList("feat"), QLists.getList("feat") , 2, 2, "subj", "_", false, "_", QLists.getList("arg0")));
        tokens.add(new CoNLL09Token(3, "ate", "_", "_", "V", "_", QLists.getList("feat"), QLists.getList("feat") , 2, 2, "v", "_", true, "ate.1", QLists.getList("_")));
        //tokens.add(new CoNLL09Token(4, "food", "_", "_", "N", "_", getList("feat"), getList("feat") , 2, 2, "obj", "_", false, "_", getList("arg1")));
        CoNLL09Sentence sent = new CoNLL09Sentence(tokens);
        
        
        CorpusStatisticsPrm csPrm = new CorpusStatisticsPrm();
        CorpusStatistics cs = new CorpusStatistics(csPrm);
        AnnoSentenceCollection sents = getSingleton(sent.toAnnoSentence(csPrm.useGoldSyntax));
        cs.init(sents);
        
        System.out.println("Done reading.");
        FactorTemplateList fts = new FactorTemplateList();
        JointNlpFgExampleBuilderPrm prm = new JointNlpFgExampleBuilderPrm();
        
        prm.fgPrm.srlPrm.roleStructure = RoleStructure.PREDS_GIVEN;
        prm.fgPrm.dpPrm.useProjDepTreeFactor = true;
        prm.fgPrm.dpPrm.linkVarType = VarType.LATENT;
        prm.fgPrm.srlPrm.srlFePrm.biasOnly = true;
        
        ObsFeatureConjoiner ofc = new ObsFeatureConjoiner(new ObsFeatureConjoinerPrm(), fts);
        JointNlpFgExamplesBuilder builder = new JointNlpFgExamplesBuilder(prm, ofc, cs);
        FgExampleList data = builder.getData(sents);
        
        LFgExample ex = data.get(0);
        
        // Global factor should still be there.
        assertEquals(1 + 3 + 3*2 + 2 + 2, ex.getFactorGraph().getFactors().size());
        // Includes an extra 2 ClampFactors
        FactorGraph fgLat = MarginalLogLikelihood.getFgLat(ex.getFactorGraph(), ex.getGoldConfig());                
        assertEquals(1 + 3 + 3*2 + 2 + 2 + 2, fgLat.getFactors().size());

        // We no longer use empty factors.
        assertEquals(0, getEmpty(ex.getFactorGraph().getFactors()).size());
        assertEquals(0, getEmpty(fgLat.getFactors()).size());
    }

    /** Gets the factors containing zero variables. */
    private List<Factor> getEmpty(List<Factor> factors) {
        ArrayList<Factor> filt = new ArrayList<Factor>();
        for (Factor f : factors) {
            if (f.getVars().size() == 0) {
                filt.add(f);
            }
        }
        return filt;
    }
    
    
        
}
