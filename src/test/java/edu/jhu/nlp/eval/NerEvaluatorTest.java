package edu.jhu.nlp.eval;

import static org.junit.Assert.assertEquals;

import java.util.List;

import org.junit.Test;

import edu.jhu.hlt.util.JUnitUtils;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.data.simple.AnnoSentenceCollection;
import edu.jhu.nlp.eval.SrlEvaluator.SrlEvaluatorPrm;
import edu.jhu.pacaya.util.collections.QLists;
import edu.jhu.pacaya.util.report.ReporterManager;

public class NerEvaluatorTest {

    static {
        ReporterManager.init(null, true);        
    }
    
    @Test
    public void testZeros() {
        List<String> gold = QLists.getList();
        List<String> pred = QLists.getList();
        evaluate(gold, pred, 0, 0, 0, 0);
    }

    @Test
    public void testMissing() {
        List<String> gold = QLists.getList("B-PER", "B-ORG", "I-ORG", "I-ORG", "O", "B-LOC");
        List<String> pred = null;
        evaluate(gold, pred, 0, 0, 3, 1);
    }
    
    @Test
    public void testCorrect() {
        List<String> gold = QLists.getList("B-PER", "B-ORG", "I-ORG", "I-ORG", "O", "B-LOC");
        List<String> pred = QLists.getList("B-PER", "B-ORG", "I-ORG", "I-ORG", "O", "B-LOC");
        evaluate(gold, pred, 3, 3, 3, 0);
    }
    
    @Test
    public void testIncorrect1() {
        List<String> gold = QLists.getList("B-PER", "B-ORG", "I-ORG", "I-ORG", "O", "B-LOC");
        List<String> pred = QLists.getList("B-PER", "B-ORG", "I-ORG", "O", "O", "B-LOC");
        evaluate(gold, pred, 2, 3, 3, 0);
    }
    
    @Test
    public void testIncorrect2() {
        List<String> gold = QLists.getList("B-PER", "B-ORG", "I-ORG", "I-ORG", "O", "B-LOC");
        List<String> pred = QLists.getList("O", "B-ORG", "I-ORG", "O", "O", "B-LOC");
        evaluate(gold, pred, 1, 2, 3, 0);
    }

    private static void evaluate(List<String> gold, List<String> pred, int numCorrectPositives, int numPredictedPositives, int numTruePositives, 
            int numMissing) {
        AnnoSentence goldSent = new AnnoSentence();
        goldSent.setNeTags(gold);
        AnnoSentenceCollection goldSents = new AnnoSentenceCollection();
        goldSents.add(goldSent);
        
        AnnoSentence predSent = new AnnoSentence();
        predSent.setNeTags(pred);
        AnnoSentenceCollection predSents = new AnnoSentenceCollection();
        predSents.add(predSent);
        
        NerEvaluator eval = new NerEvaluator();
        eval.evaluate(predSents, goldSents, "eval");
        
        double ep = numPredictedPositives == 0 ? 0 : (double) numCorrectPositives / numPredictedPositives;
        double er = numTruePositives == 0 ? 0 : (double) numCorrectPositives / numTruePositives;
        double ef1 = (ep + er) == 0 ? 0 : 2 * ep * er / (ep + er);

        assertEquals(ep, eval.getPrecision(), 1e-13);
        assertEquals(er, eval.getRecall(), 1e-13);
        assertEquals(ef1, eval.getF1(), 1e-13);
        assertEquals(numMissing, eval.getNumMissing());
    }
    
    @Test
    public void testBioTagsToChunksBIO() throws Exception {
        List<String> tags = QLists.getList("B-PER", "B-ORG", "I-ORG", "I-ORG", "O", "B-LOC");
        String[][] chunks = NerEvaluator.bioTagsToChunks(tags);
        String[][] expected = new String[6+1][6+1];
        expected[0][1] = "PER";
        expected[1][4] = "ORG";
        expected[5][6] = "LOC";
        JUnitUtils.assertArrayEquals(expected, chunks);
    }

    @Test
    public void testBioTagsToChunksIO() throws Exception {
        List<String> tags = QLists.getList("I-PER", "I-ORG", "I-ORG", "I-ORG", "O", "I-LOC");
        String[][] chunks = NerEvaluator.bioTagsToChunks(tags);
        String[][] expected = new String[6+1][6+1];
        expected[0][1] = "PER";
        expected[1][4] = "ORG";
        expected[5][6] = "LOC";
        JUnitUtils.assertArrayEquals(expected, chunks);
    }

    @Test
    public void testBioTagsToChunksBIEO() throws Exception {
        List<String> tags = QLists.getList("B-PER", "B-ORG", "I-ORG", "E-ORG", "O", "B-LOC");
        String[][] chunks = NerEvaluator.bioTagsToChunks(tags);
        String[][] expected = new String[6+1][6+1];
        expected[0][1] = "PER";
        expected[1][4] = "ORG";
        expected[5][6] = "LOC";
        JUnitUtils.assertArrayEquals(expected, chunks);
    }

    @Test
    public void testBioTagsToChunksBIEOS() throws Exception {
        List<String> tags = QLists.getList("B-PER", "B-ORG", "I-ORG", "E-ORG", "O", "S-LOC", "S-LOC");
        String[][] chunks = NerEvaluator.bioTagsToChunks(tags);
        String[][] expected = new String[7+1][7+1];
        expected[0][1] = "PER";
        expected[1][4] = "ORG";
        expected[5][6] = "LOC";
        expected[6][7] = "LOC";
        JUnitUtils.assertArrayEquals(expected, chunks);
    }

    @Test
    public void testBioTagsToChunksBIEOU() throws Exception {
        List<String> tags = QLists.getList("B-PER", "B-ORG", "I-ORG", "E-ORG", "O", "U-LOC", "U-LOC");
        String[][] chunks = NerEvaluator.bioTagsToChunks(tags);
        String[][] expected = new String[7+1][7+1];
        expected[0][1] = "PER";
        expected[1][4] = "ORG";
        expected[5][6] = "LOC";
        expected[6][7] = "LOC";
        JUnitUtils.assertArrayEquals(expected, chunks);
    }
    
}
