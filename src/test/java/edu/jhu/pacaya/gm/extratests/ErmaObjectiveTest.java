package edu.jhu.pacaya.gm.extratests;

import static org.junit.Assert.assertEquals;

import java.io.IOException;

import org.junit.Before;
import org.junit.Test;

import edu.jhu.nlp.CorpusStatistics;
import edu.jhu.nlp.CorpusStatistics.CorpusStatisticsPrm;
import edu.jhu.nlp.data.conll.CoNLL09ReadWriteTest;
import edu.jhu.nlp.data.simple.AnnoSentenceCollection;
import edu.jhu.nlp.data.simple.AnnoSentenceReader;
import edu.jhu.nlp.data.simple.AnnoSentenceReader.AnnoSentenceReaderPrm;
import edu.jhu.nlp.data.simple.AnnoSentenceReader.DatasetType;
import edu.jhu.nlp.joint.JointNlpFgExamplesBuilder;
import edu.jhu.nlp.joint.JointNlpFgExamplesBuilder.JointNlpFgExampleBuilderPrm;
import edu.jhu.pacaya.autodiff.ModuleTestUtils;
import edu.jhu.pacaya.autodiff.erma.ErmaBp.ErmaBpPrm;
import edu.jhu.pacaya.autodiff.erma.ErmaObjective;
import edu.jhu.pacaya.autodiff.erma.ErmaObjective.DlFactory;
import edu.jhu.pacaya.autodiff.erma.ExpectedRecall.ExpectedRecallFactory;
import edu.jhu.pacaya.autodiff.erma.MeanSquaredError.MeanSquaredErrorFactory;
import edu.jhu.pacaya.gm.data.FgExampleList;
import edu.jhu.pacaya.gm.data.FgExampleListBuilder.CacheType;
import edu.jhu.pacaya.gm.feat.FactorTemplateList;
import edu.jhu.pacaya.gm.feat.ObsFeatureConjoiner;
import edu.jhu.pacaya.gm.feat.ObsFeatureConjoiner.ObsFeatureConjoinerPrm;
import edu.jhu.pacaya.gm.inf.BeliefPropagation.BpScheduleType;
import edu.jhu.pacaya.gm.inf.BeliefPropagation.BpUpdateOrder;
import edu.jhu.pacaya.gm.model.FgModel;
import edu.jhu.pacaya.gm.model.Var.VarType;
import edu.jhu.pacaya.gm.train.AvgBatchObjective;
import edu.jhu.pacaya.gm.train.CrfObjective;
import edu.jhu.pacaya.util.semiring.Algebra;
import edu.jhu.pacaya.util.semiring.LogSignAlgebra;
import edu.jhu.pacaya.util.semiring.RealAlgebra;
import edu.jhu.prim.arrays.DoubleArrays;
import edu.jhu.prim.util.random.Prng;

public class ErmaObjectiveTest {
    
    @Before
    public void setUp() {
        Prng.seed(1l);
    }
    
    @Test
    public void testDpData() throws IOException {
        helpDpDataErma(new ExpectedRecallFactory(), RealAlgebra.REAL_ALGEBRA);
        helpDpDataErma(new MeanSquaredErrorFactory(), RealAlgebra.REAL_ALGEBRA);
        helpDpDataErma(new ExpectedRecallFactory(), LogSignAlgebra.LOG_SIGN_ALGEBRA);
        helpDpDataErma(new MeanSquaredErrorFactory(), LogSignAlgebra.LOG_SIGN_ALGEBRA);
    }

    private void helpDpDataErma(DlFactory dl, Algebra s) throws IOException {
        FactorTemplateList fts = new FactorTemplateList();
        ObsFeatureConjoiner ofc = new ObsFeatureConjoiner(new ObsFeatureConjoinerPrm(), fts);

        FgExampleList data = getDpData(ofc, 10);
        
        System.out.println("Num features: " + ofc.getNumParams());
        FgModel model = new FgModel(ofc.getNumParams());
        model.zero();
        
        ErmaObjective exObj = new ErmaObjective(data, getErmaBpPrm(s), dl);
        AvgBatchObjective obj = new AvgBatchObjective(exObj, model, 1);

        System.out.println(DoubleArrays.toString(obj.getGradient(model.getParams()).toNativeArray(), "%.4g"));
                
        model.setRandomStandardNormal();        
        ModuleTestUtils.assertGradientCorrectByFd(obj, model.getParams(), 1e-5, 1e-7);
    }

    public static ErmaBpPrm getErmaBpPrm(Algebra s) {
        ErmaBpPrm bpPrm = new ErmaBpPrm();
        bpPrm.s = s;
        bpPrm.schedule = BpScheduleType.TREE_LIKE;
        bpPrm.updateOrder = BpUpdateOrder.SEQUENTIAL;
        bpPrm.normalizeMessages = false;
        bpPrm.maxIterations = 1;        
        return bpPrm;
    }    

    // TODO: Is this redundant with tests in CrfTrainerTest?
    @Test
    public void testDpDataOnCrfObjective() throws IOException {
        FactorTemplateList fts = new FactorTemplateList();
        ObsFeatureConjoiner ofc = new ObsFeatureConjoiner(new ObsFeatureConjoinerPrm(), fts);

        FgExampleList data = getDpData(ofc, 10);
        
        System.out.println("Num features: " + ofc.getNumParams());
        FgModel model = new FgModel(ofc.getNumParams());
        model.setRandomStandardNormal();

        CrfObjective exObj = new CrfObjective(data, getErmaBpPrm(RealAlgebra.REAL_ALGEBRA));
        AvgBatchObjective obj = new AvgBatchObjective(exObj, model, 1);
        
        ModuleTestUtils.assertGradientCorrectByFd(obj, model.getParams(), 1e-5, 1e-8);
    }
    
    public static FgExampleList getDpData(ObsFeatureConjoiner ofc, int featureHashMod) throws IOException {
        AnnoSentenceReaderPrm rPrm = new AnnoSentenceReaderPrm();
        rPrm.maxNumSentences = 3;
        rPrm.maxSentenceLength = 7;
        rPrm.useCoNLLXPhead = true;
        AnnoSentenceReader r = new AnnoSentenceReader(rPrm);
        r.loadSents(ErmaObjectiveTest.class.getResourceAsStream(CrfObjectiveTest.conllXExample), DatasetType.CONLL_X);        
        
        CorpusStatisticsPrm csPrm = new CorpusStatisticsPrm();
        CorpusStatistics cs = new CorpusStatistics(csPrm);
        AnnoSentenceCollection sents = r.getData();
        assertEquals(rPrm.maxNumSentences, sents.size());
        cs.init(sents);
        
        JointNlpFgExampleBuilderPrm prm = new JointNlpFgExampleBuilderPrm();
        prm.fgPrm.includeSrl = false;
        prm.fgPrm.dpPrm.linkVarType = VarType.PREDICTED;
        prm.fgPrm.dpPrm.useProjDepTreeFactor = true;
        prm.exPrm.cacheType = CacheType.NONE;
        prm.fePrm.dpFePrm.featureHashMod = featureHashMod;
        
        JointNlpFgExamplesBuilder builder = new JointNlpFgExamplesBuilder(prm, ofc, cs);
        FgExampleList data = builder.getData(sents);
        return data;
    }
    
}
