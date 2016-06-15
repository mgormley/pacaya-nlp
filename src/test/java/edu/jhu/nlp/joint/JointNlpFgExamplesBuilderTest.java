package edu.jhu.nlp.joint;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.InputStream;
import java.util.Arrays;
import java.util.List;

import org.junit.Test;

import edu.jhu.hlt.optimize.LBFGS;
import edu.jhu.hlt.optimize.LBFGS_port.LBFGSPrm;
import edu.jhu.nlp.CorpusStatistics;
import edu.jhu.nlp.CorpusStatistics.CorpusStatisticsPrm;
import edu.jhu.nlp.data.DepGraph;
import edu.jhu.nlp.data.conll.CoNLL09ReadWriteTest;
import edu.jhu.nlp.data.conll.CoNLL09Reader;
import edu.jhu.nlp.data.conll.CoNLL09Sentence;
import edu.jhu.nlp.data.simple.AnnoSentenceCollection;
import edu.jhu.nlp.features.TemplateSets;
import edu.jhu.nlp.joint.JointNlpFgExamplesBuilder.JointNlpFgExampleBuilderPrm;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.RoleStructure;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.RoleVar;
import edu.jhu.pacaya.gm.data.FgExampleList;
import edu.jhu.pacaya.gm.data.LFgExample;
import edu.jhu.pacaya.gm.feat.FactorTemplateList;
import edu.jhu.pacaya.gm.feat.ObsFeatureConjoiner;
import edu.jhu.pacaya.gm.feat.ObsFeatureConjoiner.ObsFeatureConjoinerPrm;
import edu.jhu.pacaya.gm.inf.BeliefPropagation.BeliefPropagationPrm;
import edu.jhu.pacaya.gm.inf.BeliefPropagation.BpScheduleType;
import edu.jhu.pacaya.gm.inf.BeliefPropagation.BpUpdateOrder;
import edu.jhu.pacaya.gm.model.FgModel;
import edu.jhu.pacaya.gm.model.Var;
import edu.jhu.pacaya.gm.model.Var.VarType;
import edu.jhu.pacaya.gm.model.VarConfig;
import edu.jhu.pacaya.gm.model.globalfac.ProjDepTreeFactor;
import edu.jhu.pacaya.gm.train.CrfTrainer;
import edu.jhu.pacaya.gm.train.CrfTrainer.CrfTrainerPrm;
import edu.jhu.pacaya.parse.dep.ParentsArray;
import edu.jhu.pacaya.util.semiring.LogSemiring;

/**
 * Unit tests for {@link JointNlpFgExamplesBuilderTest}.
 * @author mgormley
 * @author mmitchell
 */
public class JointNlpFgExamplesBuilderTest {

    @Test
    public void testGetData() throws Exception {
        InputStream inputStream = this.getClass().getResourceAsStream(CoNLL09ReadWriteTest.conll2009Example);
        CoNLL09Reader cr = new CoNLL09Reader(inputStream);
        List<CoNLL09Sentence> sents = cr.readSents(1);
        
        AnnoSentenceCollection simpleSents = new AnnoSentenceCollection();
        CorpusStatisticsPrm csPrm = new CorpusStatisticsPrm();
        CorpusStatistics cs = new CorpusStatistics(csPrm);
        for (CoNLL09Sentence s : sents) {
            s.normalizeRoleNames();
            simpleSents.add(s.toAnnoSentence(csPrm.useGoldSyntax));
        }
        cs.init(simpleSents);
        
        System.out.println("Done reading.");
        FactorTemplateList fts = new FactorTemplateList();
        JointNlpFgExampleBuilderPrm prm = new JointNlpFgExampleBuilderPrm();
        
        prm.fgPrm.dpPrm.useProjDepTreeFactor = true;
        prm.fgPrm.srlPrm.srlFePrm.biasOnly = true;

        ObsFeatureConjoiner ofc = new ObsFeatureConjoiner(new ObsFeatureConjoinerPrm(), fts);
        JointNlpFgExamplesBuilder builder = new JointNlpFgExamplesBuilder(prm, ofc, cs);
        FgExampleList data = builder.getData(simpleSents);
        
//        System.out.println("Num features: " + alphabet.size());
//        FgModel model = new FgModel(alphabet);
//        model = train(model, data);
    }

    @Test
    public void testRoleTrainAssignment() throws Exception {
        FactorTemplateList fts = new FactorTemplateList();

        InputStream inputStream = this.getClass().getResourceAsStream(CoNLL09ReadWriteTest.conll2009Example);
        CoNLL09Reader cr = new CoNLL09Reader(inputStream);
        CorpusStatisticsPrm csPrm = new CorpusStatisticsPrm();
        CorpusStatistics cs = new CorpusStatistics(csPrm);
        List<CoNLL09Sentence> sents = cr.readSents(1);
        AnnoSentenceCollection simpleSents = new AnnoSentenceCollection();
        for (CoNLL09Sentence s : sents) {
            s.normalizeRoleNames();
            simpleSents.add(s.toAnnoSentence(csPrm.useGoldSyntax));
        }
        cs.init(simpleSents);        
        JointNlpFgExampleBuilderPrm prm = new JointNlpFgExampleBuilderPrm();
        prm.fgPrm.srlPrm.srlFePrm.biasOnly = true;
        //prm.includeUnsupportedFeatures = 
        prm.fgPrm.srlPrm.roleStructure = RoleStructure.PREDS_GIVEN;

        ObsFeatureConjoiner ofc = new ObsFeatureConjoiner(new ObsFeatureConjoinerPrm(), fts);
        JointNlpFgExamplesBuilder builder = new JointNlpFgExamplesBuilder(prm, ofc, cs);
        FgExampleList data = builder.getData(simpleSents);
        LFgExample ex = data.get(0);
        //assertEquals(1, obsAlphabet.size());
        //assertEquals(6*2 + 2 + 6, fts.size());

        VarConfig vc = ex.getGoldConfig();
        System.out.println(vc.toString().replace(",", "\n"));
        for (Var v : vc.getVars()) {
            RoleVar role = (RoleVar) v;
            if (role.getParent() == 2 && role.getChild() == 0) {
                assertEquals("arg0", vc.getStateName(v));
            } else if (role.getParent() == 2 && role.getChild() == 4) {
                assertEquals("arg1", vc.getStateName(v));
            } else {
                assertEquals("_", vc.getStateName(v));
            }
        }
        assertEquals(18, vc.size());
    }

    @Test
    public void testPreprocess() throws Exception {
        InputStream inputStream = this.getClass().getResourceAsStream(CoNLL09ReadWriteTest.conll2009Example);
        CoNLL09Reader cr = new CoNLL09Reader(inputStream);
        List<CoNLL09Sentence> sents = cr.readSents(1);
        CorpusStatisticsPrm csPrm = new CorpusStatisticsPrm();
        CorpusStatistics cs = new CorpusStatistics(csPrm);

        AnnoSentenceCollection simpleSents = new AnnoSentenceCollection();
        for (CoNLL09Sentence s : sents) {
            s.normalizeRoleNames();
            simpleSents.add(s.toAnnoSentence(csPrm.useGoldSyntax));
        }

        cs.init(simpleSents);
        JointNlpFgExampleBuilderPrm prm = new JointNlpFgExampleBuilderPrm();
        prm.fgPrm.srlPrm.srlFePrm.useTemplates = true;
        prm.fgPrm.srlPrm.srlFePrm.senseTemplates = TemplateSets.getBjorkelundSenseUnigramFeatureTemplates();
        prm.fgPrm.srlPrm.srlFePrm.argTemplates = TemplateSets.getBjorkelundArgUnigramFeatureTemplates();
        prm.fgPrm.srlPrm.roleStructure = RoleStructure.PREDS_GIVEN;
        
        {
            FactorTemplateList fts = new FactorTemplateList();
            ObsFeatureConjoinerPrm ofcPrm = new ObsFeatureConjoinerPrm();
            ofcPrm.featCountCutoff = 0;            
            ObsFeatureConjoiner ofc = new ObsFeatureConjoiner(ofcPrm, fts);
            JointNlpFgExamplesBuilder builder = new JointNlpFgExamplesBuilder(prm, ofc, cs);
            FgExampleList data = builder.getData(simpleSents);
            assertEquals(340, fts.getNumObsFeats());
            assertEquals(2916, ofc.getNumParams());
        }
        {
            FactorTemplateList fts = new FactorTemplateList();
            ObsFeatureConjoinerPrm ofcPrm = new ObsFeatureConjoinerPrm();
            ofcPrm.includeUnsupportedFeatures = true;
            ofcPrm.featCountCutoff = 1;   
            ObsFeatureConjoiner ofc = new ObsFeatureConjoiner(ofcPrm, fts);
            JointNlpFgExamplesBuilder builder = new JointNlpFgExamplesBuilder(prm, ofc, cs);
            FgExampleList data = builder.getData(simpleSents);
            ofc.init(data);
            assertEquals(2916, ofc.getNumParams());            
        }
        {
            FactorTemplateList fts = new FactorTemplateList();
            ObsFeatureConjoinerPrm ofcPrm = new ObsFeatureConjoinerPrm();
            ofcPrm.includeUnsupportedFeatures = true;
            ofcPrm.featCountCutoff = -1;   
            ObsFeatureConjoiner ofc = new ObsFeatureConjoiner(ofcPrm, fts);
            JointNlpFgExamplesBuilder builder = new JointNlpFgExamplesBuilder(prm, ofc, cs);
            FgExampleList data = builder.getData(simpleSents);
            ofc.init(data);
            assertEquals(4299, ofc.getNumParams());            
        }
        {
            FactorTemplateList fts = new FactorTemplateList();
            ObsFeatureConjoinerPrm ofcPrm = new ObsFeatureConjoinerPrm();
            ofcPrm.featCountCutoff = 5;            
            ObsFeatureConjoiner ofc = new ObsFeatureConjoiner(ofcPrm, fts);
            JointNlpFgExamplesBuilder builder = new JointNlpFgExamplesBuilder(prm, ofc, cs);
            FgExampleList data = builder.getData(simpleSents);
            ofc.init(data);
            assertEquals(2349, ofc.getNumParams());
            // These are the old counts from when we used to filter based on the counting versions of 
            // FeatureTemplateList.
            //assertEquals(313, fts.getNumObsFeats());
            //assertEquals(715, ofc.getNumParams());
        }
    }

    @Test
    public void testLinkTrainAssignment() throws Exception {
        FactorTemplateList fts = new FactorTemplateList();

        InputStream inputStream = this.getClass().getResourceAsStream(CoNLL09ReadWriteTest.conll2009Example);
        CoNLL09Reader cr = new CoNLL09Reader(inputStream);
        List<CoNLL09Sentence> sents = cr.readSents(1);
        CorpusStatisticsPrm csPrm = new CorpusStatisticsPrm();
        csPrm.useGoldSyntax = true;
        
        AnnoSentenceCollection simpleSents = new AnnoSentenceCollection();
        for (CoNLL09Sentence s : sents) {
            s.normalizeRoleNames();
            simpleSents.add(s.toAnnoSentence(csPrm.useGoldSyntax));
        }
        
        CorpusStatistics cs = new CorpusStatistics(csPrm);
        cs.init(simpleSents);        
        
        JointNlpFgExampleBuilderPrm prm = new JointNlpFgExampleBuilderPrm();
        prm.fgPrm.srlPrm.srlFePrm.biasOnly = true;
        prm.fgPrm.srlPrm.roleStructure = RoleStructure.PREDS_GIVEN;
        prm.fgPrm.dpPrm.linkVarType = VarType.PREDICTED;

        ObsFeatureConjoiner ofc = new ObsFeatureConjoiner(new ObsFeatureConjoinerPrm(), fts);
        JointNlpFgExamplesBuilder builder = new JointNlpFgExamplesBuilder(prm, ofc, cs);
        FgExampleList data = builder.getData(simpleSents);
        LFgExample ex = data.get(0);
        
        VarConfig vc = ex.getGoldConfig();
        System.out.println(vc.toString().replace(",", "\n"));

        assertEquals(18 + 19*18 + 19, vc.size());
        
        int[] parents = ProjDepTreeFactor.getParents(sents.get(0).size(), vc);
        System.out.println(Arrays.toString(parents));
        assertTrue(ParentsArray.isProjective(parents));
        assertArrayEquals(new int[]{2, 2, -1, 4, 2, 4, 7, 5, 7, 8, 7, 14, 11, 11, 10, 14, 17, 15, 2}, parents);
    }

    @Test
    public void testSenseTrainAssignment() throws Exception {
        FactorTemplateList fts = new FactorTemplateList();

        InputStream inputStream = this.getClass().getResourceAsStream(CoNLL09ReadWriteTest.conll2009Example);
        CoNLL09Reader cr = new CoNLL09Reader(inputStream);
        List<CoNLL09Sentence> sents = cr.readSents(1);
        CorpusStatisticsPrm csPrm = new CorpusStatisticsPrm();
        csPrm.useGoldSyntax = true;
        
        AnnoSentenceCollection simpleSents = new AnnoSentenceCollection();
        for (CoNLL09Sentence s : sents) {
            s.normalizeRoleNames();
            simpleSents.add(s.toAnnoSentence(csPrm.useGoldSyntax));
        }
        
        CorpusStatistics cs = new CorpusStatistics(csPrm);
        cs.init(simpleSents);        
        
        JointNlpFgExampleBuilderPrm prm = new JointNlpFgExampleBuilderPrm();
        prm.fgPrm.srlPrm.srlFePrm.biasOnly = true;
        prm.fgPrm.srlPrm.roleStructure = RoleStructure.PREDS_GIVEN;
        prm.fgPrm.srlPrm.predictSense = true;
        prm.fgPrm.srlPrm.predictPredPos = false;

        ObsFeatureConjoiner ofc = new ObsFeatureConjoiner(new ObsFeatureConjoinerPrm(), fts);
        JointNlpFgExamplesBuilder builder = new JointNlpFgExamplesBuilder(prm, ofc, cs);
        FgExampleList data = builder.getData(simpleSents);
        LFgExample ex = data.get(0);
        
        VarConfig vc = ex.getGoldConfig();
        System.out.println(vc.toString().replace(",", "\n"));

        // 18 role vars and 1 sense.
        assertEquals(18+1, vc.size());
        SrlFactorGraphBuilder srlBuilder = ((JointNlpFactorGraph) ex.getFactorGraph()).getSrlBuilder();
        DepGraph srlGraph = srlBuilder.getSrlGraphFromMbrVarConfig(vc);
        assertEquals("fer.a2", srlGraph.get(-1, 2));
    }
   
    
    private static FgModel train(FgModel model, FgExampleList data) {
        BeliefPropagationPrm bpPrm = new BeliefPropagationPrm();
        bpPrm.s = LogSemiring.getInstance();
        bpPrm.schedule = BpScheduleType.TREE_LIKE;
        bpPrm.updateOrder = BpUpdateOrder.SEQUENTIAL;
        bpPrm.normalizeMessages = false;
        
        CrfTrainerPrm prm = new CrfTrainerPrm();
        prm.infFactory = bpPrm;
        prm.optimizer = new LBFGS(new LBFGSPrm());
        // To run with SGD, uncomment these lines.
        //        SGDPrm optPrm = new SGDPrm();
        //        optPrm.iterations = 100;
        //        optPrm.lrAtMidpoint = 0.1;
        //        prm.maximizer = new SGD(optPrm);
        
        CrfTrainer trainer = new CrfTrainer(prm);
        trainer.train(model, data);
        return model;
    }
}
