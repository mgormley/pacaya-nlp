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

public class RelationEvaluatorTest {

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
        {
            // Sentence 1.
            AnnoSentence pred = new AnnoSentence();
            AnnoSentence gold = new AnnoSentence();
            
            pred.setRelLabels(QLists.getList(
                    "Other",
                    "Instrument-Agency(e2,e1)",
                    "Other",
                    "Other",
                    "Other",
                    "Other",
                    "Other",
                    "Entity-Destination(e1,e2)",
                    "Content-Container(e1,e2)",
                    "Entity-Destination(e1,e2)",
                    "Member-Collection(e1,e2)",
                    "Other",
                    "Message-Topic(e1,e2)",
                    "Cause-Effect(e2,e1)",
                    "Instrument-Agency(e2,e1)",
                    "Message-Topic(e1,e2)",
                    "Instrument-Agency(e2,e1)",
                    "Product-Producer(e2,e1)",
                    "Component-Whole(e1,e2)",
                    "Member-Collection(e2,e1)",
                    "Entity-Origin(e1,e2)",
                    "Other",
                    "Cause-Effect(e1,e2)",
                    "Member-Collection(e2,e1)",
                    "Member-Collection(e2,e1)",
                    "Other",
                    "Cause-Effect(e2,e1)",
                    "Message-Topic(e2,e1)",
                    "Message-Topic(e1,e2)",
                    "Component-Whole(e1,e2)"));
            gold.setRelLabels(QLists.getList(
                    "Component-Whole(e2,e1)",
                    "Other",
                    "Instrument-Agency(e2,e1)",
                    "Other",
                    "Member-Collection(e1,e2)",
                    "Other",
                    "Cause-Effect(e2,e1)",
                    "Entity-Destination(e1,e2)",
                    "Content-Container(e1,e2)",
                    "Entity-Destination(e1,e2)",
                    "Member-Collection(e1,e2)",
                    "Other",
                    "Message-Topic(e1,e2)",
                    "Cause-Effect(e2,e1)",
                    "Instrument-Agency(e2,e1)",
                    "Message-Topic(e1,e2)",
                    "Instrument-Agency(e2,e1)",
                    "Product-Producer(e2,e1)",
                    "Component-Whole(e2,e1)",
                    "Member-Collection(e2,e1)",
                    "Entity-Origin(e1,e2)",
                    "Member-Collection(e2,e1)",
                    "Cause-Effect(e1,e2)",
                    "Other",
                    "Member-Collection(e2,e1)",
                    "Other",
                    "Cause-Effect(e1,e2)",
                    "Message-Topic(e1,e2)",
                    "Message-Topic(e1,e2)",
                    "Component-Whole(e1,e2)"));
            
            predSents.add(pred);
            goldSents.add(gold);
        }
        {
            // Sentence 2.
            AnnoSentence pred = new AnnoSentence();
            AnnoSentence gold = new AnnoSentence();
            
            pred.setRelLabels(null);
            gold.setRelLabels(QLists.getList(
                "Message-Topic(e2,e1)",
                "Cause-Effect(e2,e1)",
                "Product-Producer(e1,e2)",
                "Entity-Destination(e1,e2)",
                "Component-Whole(e1,e2)",
                "Entity-Origin(e1,e2)",
                "Other",
                "Component-Whole(e2,e1)",
                "Cause-Effect(e1,e2)",
                "Instrument-Agency(e2,e1)"));
    
            predSents.add(pred);
            goldSents.add(gold);
        }
    }

    @Test
    public void testZeros() {
        RelationEvaluator eval = new RelationEvaluator();
        eval.evaluate(new AnnoSentenceCollection(), new AnnoSentenceCollection(), "empty dataset");
        assertEquals(0.0, eval.getPrecision(), 1e-13);
        assertEquals(0.0, eval.getRecall(), 1e-13);
        assertEquals(0.0, eval.getF1(), 1e-13);
    }
    
    @Test
    public void testUnlabeled() {
        checkPrecRecallF1(16, 21, 33, 1);
    }

    protected void checkPrecRecallF1(int numCorrectPositives, int numPredictedPositives, int numTruePositives, 
            int numMissing) {
        double ep = numPredictedPositives == 0 ? 0 : (double) numCorrectPositives / numPredictedPositives;
        double er = numTruePositives == 0 ? 0 : (double) numCorrectPositives / numTruePositives;
        double ef1 = (ep + er) == 0 ? 0 : 2 * ep * er / (ep + er);
        RelationEvaluator eval = new RelationEvaluator();
        eval.evaluate(predSents, goldSents, "Train");
        assertEquals(ep, eval.getPrecision(), 1e-13);
        assertEquals(er, eval.getRecall(), 1e-13);
        assertEquals(ef1, eval.getF1(), 1e-13);
        assertEquals(numMissing, eval.getNumMissing());
    }

}
