package edu.jhu.nlp.sprl;

import static org.junit.Assert.assertEquals;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.junit.Test;

import edu.jhu.nlp.data.concrete.ConcreteReader;
import edu.jhu.nlp.data.concrete.ConcreteReader.ConcreteReaderPrm;
import edu.jhu.nlp.data.simple.AnnoSentenceCollection;
import edu.jhu.nlp.data.simple.CorpusHandler;
import edu.jhu.nlp.eval.SprlEvaluator;
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
        SprlClassLabel.splitMode = SprlClassLabel.SplitMode.Split_123_45;
        SprlClassLabel.modelNA = false;
        SprlClassLabel.sprlNils = "NA,NOT_AN_ARG,UNKNOWN,UNLIKELY";
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
        assertEquals(-1.0, eval.evaluate(sents, sents, "gold"), 1E-6);
        assertEquals(2, eval.getNumCorrectPositive());
        assertEquals(4, eval.getNumCorrectNegative());
    }
}
