package edu.jhu.nlp.sprl;

import static org.junit.Assert.assertEquals;

import java.io.File;
import java.io.IOException;

import org.junit.Test;

import edu.jhu.nlp.data.concrete.ConcreteReader;
import edu.jhu.nlp.data.concrete.ConcreteReader.ConcreteReaderPrm;
import edu.jhu.nlp.data.simple.AnnoSentenceCollection;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.RoleStructure;
import edu.jhu.pacaya.util.report.ReporterManager;

public class SprlEvaluatorTest {
    private static String concreteFilename = "/edu/jhu/nlp/data/concrete/sprlexample.comm";
    static {   
        ReporterManager.init(ReporterManager.reportOut, true);
    }

    @Test
    public void testCorrectSprl() throws IOException {
        File f = new File(getClass().getResource(concreteFilename).getFile());
        ConcreteReaderPrm prm = new ConcreteReaderPrm();
        prm.srlTool = "fpropbank";
        prm.sprlTool = "fpropbank";
        ConcreteReader r = new ConcreteReader(prm);
        AnnoSentenceCollection sents = r.sentsFromCommFile(f);
        assertEquals(9, sents.size());
        assertEquals(0, sents.get(0).getSprl().getPairs().size()); 
        assertEquals(0, sents.get(1).getSprl().getPairs().size()); 
        assertEquals(0, sents.get(2).getSprl().getPairs().size()); 
        assertEquals(2, sents.get(3).getSprl().getPairs().size()); 
        assertEquals(1, sents.get(4).getSprl().getPairs().size()); 
        assertEquals(0, sents.get(5).getSprl().getPairs().size()); 
        assertEquals(0, sents.get(6).getSprl().getPairs().size()); 
        assertEquals(0, sents.get(7).getSprl().getPairs().size()); 
        assertEquals(0, sents.get(8).getSprl().getPairs().size()); 
        
        SprlEvaluator eval = new SprlEvaluator(RoleStructure.PAIRS_GIVEN, true, true, true, true);
        eval.evaluate(sents, sents, "gold");
        ConfusionMap<String, String> cms = eval.getConfusions();
        assertEquals(1.0, cms.getTotal().f1(), 1E-6);
        assertEquals(2, cms.getTotal().getCorrectHits());
        assertEquals(2, cms.getTotal().getNumPossible());
        assertEquals(4, cms.getTotal().getCorrectNils());

        SprlEvaluator eval2 = new SprlEvaluator(RoleStructure.PAIRS_GIVEN, true, false, false, false);
        eval2.evaluate(sents, sents, "gold");
        cms = eval2.getConfusions();
        assertEquals(1.0, cms.getTotal().f1(), 1E-6);
        assertEquals(2, cms.getTotal().getCorrectHits());
        assertEquals(2, cms.getTotal().getNumPossible());
        assertEquals(4, cms.getTotal().getCorrectNils());

        AnnoSentenceCollection sentsBad = r.sentsFromCommFile(f);
        SprlProperties sprl = sentsBad.get(3).getSprl();
        sprl.set(1,  0,  "awareness", sprl.getConverter().toLabel(2.0, true));
        SprlEvaluator eval3 = new SprlEvaluator(RoleStructure.PAIRS_GIVEN, true, true, true, true);
        eval3.evaluate(sentsBad, sents, "bad");
        cms = eval3.getConfusions();
        assertEquals(1, cms.getTotal().getCorrectHits());
        assertEquals(2, cms.getTotal().getNumPossible());
        assertEquals(1, cms.getTotal().getNumPositive());
        assertEquals(4, cms.getTotal().getCorrectNils());

    }
}
