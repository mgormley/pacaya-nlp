package edu.jhu.nlp.eval;

import static org.junit.Assert.assertEquals;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.junit.Before;
import org.junit.Test;

import edu.jhu.nlp.data.Properties;
import edu.jhu.nlp.data.Properties.Property;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.data.simple.AnnoSentenceCollection;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.RoleStructure;
import edu.jhu.pacaya.util.collections.QLists;
import edu.jhu.pacaya.util.report.ReporterManager;
import edu.jhu.prim.set.IntHashSet;
import edu.jhu.prim.tuple.Pair;

public class SprlRMSEEvaluatorTest {

    AnnoSentenceCollection predSents;
    AnnoSentenceCollection goldSents;

    static {
        ReporterManager.init(null, true);        
    }
    
    /**
     * Creates two SRL graphs (predicted and gold) and stores them in AnnoSentenceCollections.
     */
    @Before
    public void testEval() {
        predSents = new AnnoSentenceCollection();
        goldSents = new AnnoSentenceCollection();
        AnnoSentence pred = new AnnoSentence();
        AnnoSentence gold = new AnnoSentence();
        int n = 4;
        List<String> words = QLists.getList("cats", "like", "eating", "food");
        pred.setWords(words);
        gold.setWords(words);
        Map<Pair<Integer, Integer>, Properties> goldSprl = new HashMap<>();
        Map<Pair<Integer, Integer>, Properties> predSprl = new HashMap<>();
        // make a sentence with 1 pred and 2 args
        Properties predProps = new Properties();
        Properties goldProps = new Properties();
        for (Property q : Property.values()) {
            predProps.add(q.name(), 4.0);
            goldProps.add(q.name(), 4.0);
        }
        IntHashSet knownPreds = new IntHashSet();
        knownPreds.add(1);
        gold.setKnownPreds(knownPreds);
        pred.setKnownPreds(knownPreds);
        Set<Pair<Integer, Integer>> knownPairs = new HashSet<>();
        knownPairs.add(new Pair<>(1,1));
        knownPairs.add(new Pair<>(1,3));
        gold.setKnownSrlPairs(knownPairs);
        pred.setKnownSrlPairs(knownPairs);
        predProps.add(Property.change_of_location.name(), 2.0);
        predProps.add(Property.change_of_state.name(), 5.0);        

        goldSprl.put(new Pair<Integer, Integer>(1, 1), goldProps);
        goldSprl.put(new Pair<Integer, Integer>(1, 3), goldProps); 
        predSprl.put(new Pair<Integer, Integer>(1, 1), predProps);  // some wrong things
        predSprl.put(new Pair<Integer, Integer>(1, 2), goldProps);  // should be nil
        predSprl.put(new Pair<Integer, Integer>(1, 3), goldProps);  // these are correct      
        gold.setSprl(goldSprl);
        pred.setSprl(predSprl);

        predSents.add(pred);
        goldSents.add(gold);

        // not skipping nils
        {
            SprlRMSEEvaluator eval = new SprlRMSEEvaluator(RoleStructure.PREDS_GIVEN, true, false);
            int numExamples = 4 * Properties.nquestions; 
            double expectedRMSE = Math.sqrt(
                    (Properties.nquestions * (0.75 * 0.75) + // for the extra sprl at 1,2 where they should be 0's 
                            (.75 - .25) * (.75 - .25) + (.75 - 1.0) * (.75 - 1.0)) // for the two incorrect labels on 1,1
                    / numExamples // divided by the total questions for the 4 possible pairs 
                    );
            assertEquals("match rmse", expectedRMSE, eval.evaluate(predSents, goldSents, "SPRL_RMSE"), 1E-13);
            assertEquals("match sotred rmse", expectedRMSE, eval.getRMSE(), 1E-13);
            assertEquals("num examples", eval.getNumExamples(), numExamples, 1E-13);
            assertEquals("num examples missing", eval.getNumExamplesMissing(), 0, 1E-13);
            assertEquals("num structures", eval.getNumStructures(), 1, 1E-13);
            assertEquals("num structures missing", eval.getNumStructuresMissing(), 0, 1E-13);
        }
        // skipping nils
        {
            SprlRMSEEvaluator eval = new SprlRMSEEvaluator(RoleStructure.PREDS_GIVEN, true, true);
            int numExamples = 4 * Properties.nquestions; 
            int numExamplesMissing = 2 * Properties.nquestions; 
            double expectedRMSE = Math.sqrt(
                    ((.75 - .25) * (.75 - .25) + (.75 - 1.0) * (.75 - 1.0)) // for the two incorrect labels on 1,1
                    / (numExamples - numExamplesMissing) // divided by the total questions for the 4 possible pairs 
                    );
            assertEquals("match rmse", expectedRMSE, eval.evaluate(predSents, goldSents, "SPRL_RMSE"), 1E-13);
            assertEquals("match sotred rmse", expectedRMSE, eval.getRMSE(), 1E-13);
            assertEquals("num examples", eval.getNumExamples(), numExamples, 1E-13);
            assertEquals("num examples missing", eval.getNumExamplesMissing(), numExamplesMissing, 1E-13);
            assertEquals("num structures", eval.getNumStructures(), 1, 1E-13);
            assertEquals("num structures missing", eval.getNumStructuresMissing(), 0, 1E-13);
        }
        // known pairs
        {
            SprlRMSEEvaluator eval = new SprlRMSEEvaluator(RoleStructure.PAIRS_GIVEN, true, false);
            int numExamples = 2 * Properties.nquestions; 
            int numExamplesMissing = 0;
            double expectedRMSE = Math.sqrt(
                    ((.75 - .25) * (.75 - .25) + (.75 - 1.0) * (.75 - 1.0)) // for the two incorrect labels on 1,1
                    / (numExamples - numExamplesMissing) // divided by the total questions for the 4 possible pairs 
                    );
            assertEquals("match rmse", expectedRMSE, eval.evaluate(predSents, goldSents, "SPRL_RMSE"), 1E-13);
            assertEquals("match sotred rmse", expectedRMSE, eval.getRMSE(), 1E-13);
            assertEquals("num examples", eval.getNumExamples(), numExamples, 1E-13);
            assertEquals("num examples missing", eval.getNumExamplesMissing(), numExamplesMissing, 1E-13);
            assertEquals("num structures", eval.getNumStructures(), 1, 1E-13);
            assertEquals("num structures missing", eval.getNumStructuresMissing(), 0, 1E-13);
        }

    }

    @Test
    public void testZeros() {
        SprlRMSEEvaluator eval = new SprlRMSEEvaluator(RoleStructure.PREDS_GIVEN, true, false);
        eval.evaluate(new AnnoSentenceCollection(), new AnnoSentenceCollection(), "empty dataset");
        assertEquals(0.0, eval.getRMSE(), 1e-13);
    }
    
}