package edu.jhu.nlp.eval;

import static org.junit.Assert.assertEquals;

import java.util.List;

import org.junit.Before;
import org.junit.Test;

import edu.jhu.nlp.data.DepGraph;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.data.simple.AnnoSentenceCollection;
import edu.jhu.nlp.eval.SrlEvaluator.SrlEvaluatorPrm;
import edu.jhu.pacaya.util.collections.QLists;
import edu.jhu.pacaya.util.report.ReporterManager;

public class SrlEvaluatorTest {

    AnnoSentenceCollection predSents;
    AnnoSentenceCollection goldSents;

    static {
        ReporterManager.init(null, true);        
    }
    
    /**
     * Creates two SRL graphs (predicted and gold) and stores them in AnnoSentenceCollections.
     */
    @Before
    public void setUp() {
        predSents = new AnnoSentenceCollection();
        goldSents = new AnnoSentenceCollection();
        AnnoSentence pred = new AnnoSentence();
        AnnoSentence gold = new AnnoSentence();
        int n = 5;
        List<String> words = QLists.getList("cats", "like", "eating", "food");
        pred.setWords(words);
        gold.setWords(words);
        
        DepGraph predSrl = new DepGraph(n);
        DepGraph goldSrl = new DepGraph(n);
        
        predSrl.set(-1, 1, "like.01"); // Pred
        predSrl.set(1, 0, "agent");    // Arg
        predSrl.set(1, 1, "patient");  // Arg (incorrect position)
        predSrl.set(1, 3, "nonarg");   // Arg
        predSrl.set(-1, 2, "drink.01");// Pred (incorrect label)
                                       // Arg (incorrect: missing 2, 0, agent.)
        predSrl.set(2, 3, "theme");    // Arg (incorrect label)
        predSrl.set(-1, 0, "run.02");  // Pred (extra)
        pred.setSrlGraph(predSrl);
        pred.setKnownPredsFromSrlGraph();
        
        goldSrl.set(-1, 1, "like.01"); // Pred
        goldSrl.set(1, 0, "agent");    // Arg
        goldSrl.set(1, 2, "patient");  // Arg
        goldSrl.set(1, 3, "nonarg");   // Arg
        goldSrl.set(-1, 2, "eat.01");  // Pred
        goldSrl.set(2, 0, "agent");    // Arg
        goldSrl.set(2, 3, "patient");  // Arg
        gold.setSrlGraph(goldSrl);
        gold.setKnownPredsFromSrlGraph();

        System.out.println(pred.getSrlGraph());
        System.out.println(gold.getSrlGraph());
        
        predSents.add(pred);
        goldSents.add(gold);
    }

    @Test
    public void testZeros() {
        SrlEvaluatorPrm prm = new SrlEvaluatorPrm();
        prm.labeled = true;
        prm.evalPredSense = true;
        prm.evalPredPosition = true;
        SrlEvaluator eval = new SrlEvaluator(prm);
        eval.evaluate(new AnnoSentenceCollection(), new AnnoSentenceCollection(), "empty dataset");
        assertEquals(0.0, eval.getPrecision(), 1e-13);
        assertEquals(0.0, eval.getRecall(), 1e-13);
        assertEquals(0.0, eval.getF1(), 1e-13);
    }

    @Test
    public void testMissingPredictions() {
        predSents.get(0).setSrlGraph(null);
        checkSrlPrecRecallF1(0, 0, 0, 1, false, false, false, true);
    }
    
    @Test
    public void testUnlabeled() {
        checkSrlPrecRecallF1(3, 4, 5, 0, false, false, false, true);
    }
    
    @Test
    public void testLabeled() {     
        checkSrlPrecRecallF1(2, 4, 5, 0, true, false, false, true);
    }
    
    @Test
    public void testUnlabeledSense() {
        checkSrlPrecRecallF1(5, 6, 7, 0, false, true, false, true);
    }
    
    @Test
    public void testLabeledSense() {     
        checkSrlPrecRecallF1(3, 6, 7, 0, true, true, false, true);
    }

    @Test
    public void testUnlabeledPosition() {
        checkSrlPrecRecallF1(5, 7, 7, 0, false, false, true, true);
    }
    
    @Test
    public void testLabeledPosition() {     
        checkSrlPrecRecallF1(4, 7, 7, 0, true, false, true, true);
    }

    @Test
    public void testUnlabeledSensePosition() {
        checkSrlPrecRecallF1(5, 7, 7, 0, false, true, true, true);
    }
    
    @Test
    public void testLabeledSensePosition() {     
        checkSrlPrecRecallF1(3, 7, 7, 0, true, true, true, true);
    }

    @Test
    public void testPositionNoRoles() {     
        checkSrlPrecRecallF1(2, 3, 2, 0, false, false, true, false);
    }
    
    @Test
    public void testSensePositionNoRoles() {     
        checkSrlPrecRecallF1(1, 3, 2, 0, true, true, true, false);
    }

    protected void checkSrlPrecRecallF1(int numCorrectPositives, int numPredictedPositives, int numTruePositives, 
            int numMissing, boolean labeled, boolean evalSense, boolean evalPredicatePosition, boolean evalRoles) {
        double ep = numPredictedPositives == 0 ? 0 : (double) numCorrectPositives / numPredictedPositives;
        double er = numTruePositives == 0 ? 0 : (double) numCorrectPositives / numTruePositives;
        double ef1 = (ep + er) == 0 ? 0 : 2 * ep * er / (ep + er);
        SrlEvaluatorPrm prm = new SrlEvaluatorPrm();
        prm.labeled = labeled;
        prm.evalPredSense = evalSense;
        prm.evalPredPosition = evalPredicatePosition;
        prm.evalRoles = evalRoles;
        SrlEvaluator eval = new SrlEvaluator(prm);
        eval.evaluate(predSents, goldSents, "Train");
        assertEquals(ep, eval.getPrecision(), 1e-13);
        assertEquals(er, eval.getRecall(), 1e-13);
        assertEquals(ef1, eval.getF1(), 1e-13);
        assertEquals(numMissing, eval.getNumMissing());
    }

}
