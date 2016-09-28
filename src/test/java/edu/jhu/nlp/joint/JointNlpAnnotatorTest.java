package edu.jhu.nlp.joint;

import static org.junit.Assert.assertEquals;

import java.io.IOException;
import java.io.InputStream;
import java.util.List;

import org.junit.Test;

import edu.jhu.nlp.data.conll.CoNLL09ReadWriteTest;
import edu.jhu.nlp.data.conll.CoNLL09Reader;
import edu.jhu.nlp.data.conll.CoNLL09Sentence;
import edu.jhu.nlp.data.simple.AnnoSentenceCollection;
import edu.jhu.nlp.eval.SrlEvaluator;
import edu.jhu.nlp.eval.SrlEvaluator.SrlEvaluatorPrm;
import edu.jhu.nlp.features.TemplateLanguage.AT;
import edu.jhu.nlp.joint.JointNlpAnnotator.JointNlpAnnotatorPrm;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.RoleStructure;
import edu.jhu.pacaya.gm.decode.MbrDecoder.MbrDecoderPrm;
import edu.jhu.pacaya.util.collections.QLists;
import edu.jhu.pacaya.util.report.ReporterManager;

public class JointNlpAnnotatorTest {

    static {
        ReporterManager.init(null, true);
    }
    
    @Test
    public void testTrainAndAnnotateSrlOnly() throws Exception {
        AnnoSentenceCollection trainGold = getSrlSents(5, 100, true);
        AnnoSentenceCollection trainInput = trainGold.getWithAtsRemoved(QLists.getList(AT.SRL));
        
        System.out.println("Number of sents: " + trainGold.size());
        
        JointNlpAnnotatorPrm prm = new JointNlpAnnotatorPrm();
        prm.buPrm.fgPrm.includePos = false;
        prm.buPrm.fgPrm.includeDp = false;
        prm.buPrm.fgPrm.includeRel = false;
        prm.buPrm.fgPrm.includeSrl = true;
        prm.buPrm.fgPrm.srlPrm.unaryFactors = true;
        prm.buPrm.fgPrm.srlPrm.roleStructure = RoleStructure.PREDS_GIVEN;
        prm.buPrm.fgPrm.srlPrm.predictSense = true;
        prm.buPrm.fgPrm.srlPrm.predictPredPos = false;
        prm.dePrm.mbrPrm = new MbrDecoderPrm();
        JointNlpAnnotator anno = new JointNlpAnnotator(prm , null);
        anno.train(trainInput, trainGold, null, null);
        anno.annotate(trainInput);
        
        SrlEvaluatorPrm ePrm = new SrlEvaluatorPrm(true, true, false, true);
        SrlEvaluator eval = new SrlEvaluator(ePrm);
        eval.evaluate(trainInput, trainGold, "train");
        assertEquals(1.0, eval.getF1(), 1e-13);
    }

    public static AnnoSentenceCollection getGoldSrlSents() throws IOException {
        return getSrlSents(Integer.MAX_VALUE, Integer.MAX_VALUE, true);
    }
    
    public static AnnoSentenceCollection getSrlSents(int maxSents, int maxSentLen, boolean useGoldSyntax) throws IOException {
        InputStream inputStream = JointNlpAnnotatorTest.class.getResourceAsStream(CoNLL09ReadWriteTest.conll2009Example);
        CoNLL09Reader cr = new CoNLL09Reader(inputStream);
        List<CoNLL09Sentence> csents = cr.readSents(maxSents, maxSentLen);
        AnnoSentenceCollection asents = new AnnoSentenceCollection();
        for (CoNLL09Sentence s : csents) {
            asents.add(s.toAnnoSentence(useGoldSyntax));
        }
        cr.close();        
        return asents;
    }
    
}
