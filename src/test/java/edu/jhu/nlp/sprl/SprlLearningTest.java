package edu.jhu.nlp.sprl;

import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import edu.jhu.hlt.optimize.FixedStep;
import edu.jhu.hlt.optimize.MalletLBFGS;
import edu.jhu.hlt.optimize.MalletLBFGS.MalletLBFGSPrm;
import edu.jhu.hlt.optimize.SGD;
import edu.jhu.hlt.optimize.SGD.SGDPrm;
import edu.jhu.hlt.optimize.functions.L2;
import edu.jhu.nlp.data.concrete.ConcreteReader;
import edu.jhu.nlp.data.concrete.ConcreteReader.ConcreteReaderPrm;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.data.simple.AnnoSentenceCollection;
import edu.jhu.nlp.features.TemplateLanguage.AT;
import edu.jhu.nlp.joint.JointNlpAnnotator;
import edu.jhu.nlp.joint.JointNlpAnnotator.JointNlpAnnotatorPrm;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.RoleStructure;
import edu.jhu.pacaya.gm.train.CrfTrainer.CrfTrainerPrm;
import edu.jhu.prim.tuple.Pair;
import edu.jhu.prim.util.random.Prng;

public class SprlLearningTest {

    private static String concreteFilename = "/edu/jhu/nlp/data/concrete/sprlexample.comm";

    private CrfTrainerPrm getLBFGSPrm() {
        CrfTrainerPrm crfPrm = new CrfTrainerPrm();
        crfPrm.batchOptimizer = null;
        crfPrm.regularizer = null;
        MalletLBFGSPrm lbfgsPrms = new MalletLBFGSPrm();
        lbfgsPrms.maxIterations = 1;
        crfPrm.optimizer = new MalletLBFGS(lbfgsPrms);
        lbfgsPrms.numberOfCorrections = 0;
        return crfPrm;
    }
    
    private CrfTrainerPrm getSGDPrm() {
        CrfTrainerPrm crfPrm = new CrfTrainerPrm();
        SGDPrm sgdPrm = new SGDPrm();
        sgdPrm.autoSelectLr = false;
        sgdPrm.batchSize = 1;
        sgdPrm.numPasses = 10;
        sgdPrm.sched = new FixedStep(1.0);
        crfPrm.batchOptimizer = new SGD(sgdPrm);
        crfPrm.optimizer = null;
        crfPrm.regularizer = new L2(1.0 / .1);
        return crfPrm;
    }

    private CrfTrainerPrm getCrfPrm() {
        return getSGDPrm();
    }
    
    @Rule
    public TemporaryFolder tempFolder = new TemporaryFolder();

    private void testIndependent(boolean biasOnly) throws IOException {
        File f = new File(getClass().getResource(concreteFilename).getFile());
        SprlClassLabel.splitMode = SprlClassLabel.SplitMode.Split_123_45;
        SprlClassLabel.modelNA = false;
        SprlClassLabel.sprlNils = "NA,NOT_AN_ARG,UNKNOWN,UNLIKELY";
        ConcreteReaderPrm prm = new ConcreteReaderPrm();
        prm.srlTool = "fpropbank";
        prm.sprlTool = "fpropbank";
        ConcreteReader r = new ConcreteReader(prm);
        AnnoSentenceCollection goldTrain = r.sentsFromCommFile(f);
        AnnoSentence exampleSentence = goldTrain.get(4);
        exampleSentence.setKnownSrlPairs(Collections.singleton(new Pair<>(5,4)));
        exampleSentence.setKnownSprlPairs(Collections.singleton(new Pair<>(5,4)));
        goldTrain = AnnoSentenceCollection.getSingleton(exampleSentence);
        AnnoSentenceCollection inTrain = goldTrain.getWithAtsRemoved(Arrays.asList(AT.SRL, AT.SPRL));
        AnnoSentenceCollection goldDev = goldTrain.getWithAtsRemoved(Arrays.asList());
        AnnoSentenceCollection inDev = goldDev.getWithAtsRemoved(Arrays.asList(AT.SRL, AT.SPRL));
        JointNlpAnnotatorPrm jointPrm = new JointNlpAnnotatorPrm();
        jointPrm.buPrm.fgPrm.srlPrm.roleStructure = RoleStructure.PAIRS_GIVEN;
        jointPrm.buPrm.fgPrm.srlPrm.srlFePrm.biasOnly = biasOnly;
        jointPrm.buPrm.fgPrm.sprlPrm.roleStructure = RoleStructure.PAIRS_GIVEN;
        jointPrm.buPrm.fgPrm.sprlPrm.srlFePrm.biasOnly = biasOnly;
        jointPrm.buPrm.fgPrm.enforceSprlNilAgreement = false;
        jointPrm.buPrm.fgPrm.sprlPrm.extraVariablesForNilAgreement = false;
        jointPrm.buPrm.fgPrm.includeDp = false;
        jointPrm.buPrm.fgPrm.includeRel = false;
        jointPrm.buPrm.fgPrm.includePos = false;
        jointPrm.buPrm.fgPrm.includeSrl = true;
        jointPrm.buPrm.fgPrm.includeSprl = false;
        jointPrm.ofcPrm.includeUnsupportedFeatures = false;
        long seed = 5;
        double[] srlParams = null;
        {
            Prng.seed(seed);
            jointPrm.crfPrm = getCrfPrm();
            jointPrm.exportTrainToLibDAI = tempFolder.newFolder("srl").getAbsolutePath();
            JointNlpAnnotator anno = new JointNlpAnnotator(jointPrm, null);

            anno.train(inTrain, goldTrain, inDev, goldDev);
            anno.getModel().printModel(new PrintWriter(System.out));
            srlParams = anno.getModel().getParams().toNativeArray();
            System.out.println(String.join("\n", Files.readAllLines(Paths.get(jointPrm.exportTrainToLibDAI, "after", "0.fg"))));
        }
        double[] jointParams = null;
        {
            Prng.seed(seed);
            jointPrm.crfPrm = getCrfPrm();
            jointPrm.buPrm.fgPrm.includeSprl = true;
            jointPrm.exportTrainToLibDAI = tempFolder.newFolder("joint").getAbsolutePath();
            JointNlpAnnotator anno = new JointNlpAnnotator(jointPrm, null);

            anno.train(inTrain, goldTrain, inDev, goldDev);
            anno.getModel().printModel(new PrintWriter(System.out));
            jointParams = anno.getModel().getParams().toNativeArray();
            System.out.println(String.join("\n", Files.readAllLines(Paths.get(jointPrm.exportTrainToLibDAI, "after", "0.fg"))));
        }
        double[] sprlParams = null;
        {
            Prng.seed(seed);
            jointPrm.crfPrm = getCrfPrm();
            jointPrm.buPrm.fgPrm.includeSrl = false;
            jointPrm.exportTrainToLibDAI = tempFolder.newFolder("sprl").getAbsolutePath();
            JointNlpAnnotator anno = new JointNlpAnnotator(jointPrm, null);

            anno.train(inTrain, goldTrain, inDev, goldDev);
            anno.getModel().printModel(new PrintWriter(System.out));
            sprlParams = anno.getModel().getParams().toNativeArray();
            System.out.println(String.join("\n", Files.readAllLines(Paths.get(jointPrm.exportTrainToLibDAI, "after", "0.fg"))));
        }
        assertTrue(cat(jointParams).equals(cat(sprlParams, srlParams)));

    }
    
    @Test
    public void testCorrectSprl() throws IOException {
        testIndependent(true);
    }
    
    private List<Double> cat(double[]...arrays) {
        List<Double> ret = new ArrayList<>();
        for (double[] a : arrays) {
            for (double d : a) {
                ret.add(d);
            }
        }
        return ret;
    }
}
