package edu.jhu.gm.extratests;

import java.io.File;
import java.io.IOException;

import org.junit.Test;

import edu.jhu.autodiff.ModuleTestUtils;
import edu.jhu.autodiff.erma.ErmaBp.ErmaBpPrm;
import edu.jhu.autodiff.erma.ErmaObjective;
import edu.jhu.autodiff.erma.ErmaObjective.DlFactory;
import edu.jhu.autodiff.erma.ExpectedRecall.ExpectedRecallFactory;
import edu.jhu.autodiff.erma.MeanSquaredError.MeanSquaredErrorFactory;
import edu.jhu.gm.data.FgExampleList;
import edu.jhu.gm.data.FgExampleListBuilder.CacheType;
import edu.jhu.gm.feat.FactorTemplateList;
import edu.jhu.gm.feat.ObsFeatureConjoiner;
import edu.jhu.gm.feat.ObsFeatureConjoiner.ObsFeatureConjoinerPrm;
import edu.jhu.gm.inf.BeliefPropagation.BpScheduleType;
import edu.jhu.gm.inf.BeliefPropagation.BpUpdateOrder;
import edu.jhu.gm.model.FgModel;
import edu.jhu.gm.model.Var.VarType;
import edu.jhu.gm.train.AvgBatchObjective;
import edu.jhu.gm.train.CrfObjective;
import edu.jhu.nlp.CorpusStatistics;
import edu.jhu.nlp.CorpusStatistics.CorpusStatisticsPrm;
import edu.jhu.nlp.data.simple.AnnoSentenceCollection;
import edu.jhu.nlp.data.simple.AnnoSentenceReader;
import edu.jhu.nlp.data.simple.AnnoSentenceReader.AnnoSentenceReaderPrm;
import edu.jhu.nlp.data.simple.AnnoSentenceReader.DatasetType;
import edu.jhu.nlp.joint.JointNlpFgExamplesBuilder;
import edu.jhu.nlp.joint.JointNlpFgExamplesBuilder.JointNlpFgExampleBuilderPrm;
import edu.jhu.prim.arrays.DoubleArrays;
import edu.jhu.util.semiring.Algebra;
import edu.jhu.util.semiring.Algebras;

public class ErmaObjectiveTest {
        
    @Test
    public void testDpData() throws IOException {
        helpDpDataErma(new ExpectedRecallFactory(), Algebras.REAL_ALGEBRA);
        helpDpDataErma(new MeanSquaredErrorFactory(), Algebras.REAL_ALGEBRA);
        helpDpDataErma(new ExpectedRecallFactory(), Algebras.LOG_SIGN_ALGEBRA);
        helpDpDataErma(new MeanSquaredErrorFactory(), Algebras.LOG_SIGN_ALGEBRA);
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

        CrfObjective exObj = new CrfObjective(data, getErmaBpPrm(Algebras.REAL_ALGEBRA));
        AvgBatchObjective obj = new AvgBatchObjective(exObj, model, 1);
        
        ModuleTestUtils.assertGradientCorrectByFd(obj, model.getParams(), 1e-5, 1e-8);
    }
    
    public static FgExampleList getDpData(ObsFeatureConjoiner ofc, int featureHashMod) throws IOException {
        AnnoSentenceReaderPrm rPrm = new AnnoSentenceReaderPrm();
        rPrm.maxNumSentences = 3;
        rPrm.maxSentenceLength = 7;
        rPrm.useCoNLLXPhead = true;
        AnnoSentenceReader r = new AnnoSentenceReader(rPrm);
        //r.loadSents(CrfObjectiveTest.class.getResourceAsStream(CoNLL09ReadWriteTest.conll2009Example), DatasetType.CONLL_2009);
        r.loadSents(new File("/Users/mgormley/research/pacaya/data/conllx/CoNLL-X/train/data/bulgarian/bultreebank/train/bulgarian_bultreebank_train.conll"), DatasetType.CONLL_X);
        
        CorpusStatisticsPrm csPrm = new CorpusStatisticsPrm();
        CorpusStatistics cs = new CorpusStatistics(csPrm);
        AnnoSentenceCollection sents = r.getData();
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
