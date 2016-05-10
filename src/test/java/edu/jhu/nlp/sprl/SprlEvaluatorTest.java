package edu.jhu.nlp.sprl;

import static org.junit.Assert.assertEquals;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

import org.junit.Test;

import edu.jhu.nlp.data.concrete.ConcreteReader;
import edu.jhu.nlp.data.concrete.ConcreteReader.ConcreteReaderPrm;
import edu.jhu.nlp.data.simple.AnnoSentenceCollection;
import edu.jhu.nlp.eval.SprlEvaluator;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.RoleStructure;
import edu.jhu.pacaya.util.report.ReporterManager;
import edu.jhu.prim.tuple.Pair;

public class SprlEvaluatorTest {
    private static String concreteFilename = "/edu/jhu/nlp/data/concrete/sprlexample.comm";
    static {   
        ReporterManager.init(ReporterManager.reportOut, true);
    }

    @Test
    public void testCorrectSprl() throws IOException {
/*
        File f = new File(getClass().getResource(concreteFilename).getFile());
        ConcreteReaderPrm prm = new ConcreteReaderPrm();
        prm.srlTool = "fpropbank";
        prm.sprlTool = "fpropbank";
        ConcreteReader r = new ConcreteReader(prm);
        AnnoSentenceCollection sents = r.sentsFromCommFile(f);
        assertEquals(9, sents.size());
        assertEquals(0, sents.get(0).getSprl().size()); 
        assertEquals(0, sents.get(1).getSprl().size()); 
        assertEquals(0, sents.get(2).getSprl().size()); 
        assertEquals(2, sents.get(3).getSprl().size()); 
        assertEquals(1, sents.get(4).getSprl().size()); 
        assertEquals(0, sents.get(5).getSprl().size()); 
        assertEquals(0, sents.get(6).getSprl().size()); 
        assertEquals(0, sents.get(7).getSprl().size()); 
        assertEquals(0, sents.get(8).getSprl().size()); 
//        List<String> propsToScore = new ArrayList<>(CorpusHandler.getKnownSprlProperties(sents));
        SprlEvaluator eval = new SprlEvaluator(RoleStructure.PAIRS_GIVEN, true, SprlClassLabel.getNils());
        eval.evaluate(sents, sents, "gold");
        assertEquals(1.0, eval.getF1(), 1E-6);
        assertEquals(2, eval.getNumCorrectPositive());
        assertEquals(2, eval.getNumTruePositive());
        assertEquals(4, eval.getNumCorrectNegative());
        SprlEvaluator eval2 = new SprlEvaluator(RoleStructure.PAIRS_GIVEN, true, SprlClassLabel.getNils(), Arrays.asList("awareness"));
        eval2.evaluate(sents, sents, "gold");
        assertEquals(1.0, eval2.getF1(), 1E-6);
        assertEquals(2, eval2.getNumCorrectPositive());
        assertEquals(2, eval2.getNumTruePositive());
        assertEquals(1, eval2.getNumCorrectNegative());

        AnnoSentenceCollection sentsBad = r.sentsFromCommFile(f);
        sentsBad.get(3).getSprl().get(new Pair<>(1,0)).add("awareness", 2.0);
        SprlEvaluator eval3 = new SprlEvaluator(RoleStructure.PAIRS_GIVEN, true, SprlClassLabel.getNils());
        eval3.evaluate(sentsBad, sents, "bad");
        assertEquals(1, eval3.getNumCorrectPositive());
        assertEquals(2, eval3.getNumTruePositive());
        assertEquals(6, eval3.getNumInstances());
        assertEquals(0, eval3.getNumMissing());
        assertEquals(1, eval3.getNumPredictPositive());
        assertEquals(4, eval3.getNumCorrectNegative());
  */      
    }
}
