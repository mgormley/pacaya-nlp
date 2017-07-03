package edu.jhu.nlp.sprl;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.junit.Test;

import edu.jhu.pacaya.sch.util.TestUtils;
import edu.jhu.pacaya.util.report.Reporter;
import edu.jhu.prim.tuple.Pair;

public class ConfusionMapTest {
//    private static double tol = 1E-9;
    
    @Test
    public void testCounts() {
        String cat1 = "cat1";
        ConfusionMap<Integer, String> cm = new ConfusionMap<>(Collections.singleton(0));
        cm.recordPrediction(3, 3, cat1); // 1 ...
        cm.recordPrediction(0, 2, cat1); // 1
        cm.recordPrediction(0, 1, cat1); // 1 ...
        cm.recordPrediction(3, 1, cat1); // 1
        cm.recordPrediction(0, 1, cat1); // 2
        cm.recordPrediction(3, 5, cat1); // 1
        cm.recordPrediction(0, 0, cat1); // 1 ...
        cm.recordPrediction(0, 0, cat1); // 2
        cm.recordPrediction(2, 0, cat1); // 1
        cm.recordPrediction(3, 3, cat1); // 2
        assertEquals(cm.getConfusionMatrix(cat1).getCount(0, 0), 2);
        assertEquals(cm.getConfusionMatrix(cat1).getCount(0, 1), 2);
        assertEquals(cm.getConfusionMatrix(cat1).getCount(0, 2), 1);
        assertEquals(cm.getConfusionMatrix(cat1).getCount(2, 0), 1);
        assertEquals(cm.getConfusionMatrix(cat1).getCount(2, 2), 0);
        assertEquals(cm.getConfusionMatrix(cat1).getCount(3, 1), 1);
        assertEquals(cm.getConfusionMatrix(cat1).getCount(3, 3), 2);
        assertEquals(cm.getConfusionMatrix(cat1).getCount(3, 5), 1);
        assertEquals(cm.getConfusionMatrix(cat1).getCount(4, 4), 0);
        assertEquals(cm.getConfusionMatrix(cat1).getTotal(), 10);
        assertEquals(cm.getConfusionMatrix(cat1).getGoldCount(0), 5);
        assertEquals(cm.getConfusionMatrix(cat1).getGoldCount(1), 0);
        assertEquals(cm.getConfusionMatrix(cat1).getGoldCount(2), 1);
        assertEquals(cm.getConfusionMatrix(cat1).getGoldCount(3), 4);
        assertEquals(cm.getConfusionMatrix(cat1).getGoldCount(4), 0);
        assertEquals(cm.getConfusionMatrix(cat1).getGoldCount(5), 0);
        assertEquals(cm.getConfusionMatrix(cat1).getPredCount(0), 3);
        assertEquals(cm.getConfusionMatrix(cat1).getPredCount(1), 3);
        assertEquals(cm.getConfusionMatrix(cat1).getPredCount(2), 1);
        assertEquals(cm.getConfusionMatrix(cat1).getPredCount(3), 2);
        assertEquals(cm.getConfusionMatrix(cat1).getPredCount(5), 1);
        assertEquals(cm.getConfusionMatrix(cat1).getPredCount(6), 0);
        assertEquals(cm.getConfusionMatrix(cat1).getCorrect(), 4);
        Pair<Integer, Integer> baseline = cm.categorySpecificMaxF1Baseline();
        assertEquals(4, (int)baseline.get1());
        assertEquals(10, (int)baseline.get2());
        assertEquals((int) cm.getConfusionMatrix(cat1).majorityNonNilCorrectHits(), (int)baseline.get1());
        assertEquals((int) cm.getConfusionMatrix(cat1).getTotal(), (int)baseline.get2());

        // prec / recall
        // 4 / 10 * 4 / 8
        // 6 / 20 * 6 / 8
        String cat2 = "cat2";
        cm.recordPrediction(0, 3, cat2); // 1 ...
        cm.recordPrediction(0, 2, cat2); // 1
        cm.recordPrediction(1, 1, cat2); // 1 ...
        cm.recordPrediction(0, 1, cat2); // 1
        cm.recordPrediction(0, 1, cat2); // 2
        cm.recordPrediction(2, 5, cat2); // 1
        cm.recordPrediction(0, 0, cat2); // 1 ...
        cm.recordPrediction(0, 0, cat2); // 2
        cm.recordPrediction(2, 0, cat2); // 1
        cm.recordPrediction(0, 3, cat2); // 2

        baseline = cm.categorySpecificMaxF1Baseline();
        assertEquals(4, (int)baseline.get1());
        assertEquals(10, (int)baseline.get2());
        assertEquals((int) cm.getConfusionMatrix(cat1).majorityNonNilCorrectHits(), (int)baseline.get1());
        assertEquals((int) cm.getConfusionMatrix(cat1).getTotal(), (int)baseline.get2());

        // prec / recall
        // 4 / 10 * 4 / 8
        // 7 / 20 * 7 / 8
        String cat3 = "cat3";
        cm.recordPrediction(0, 3, cat3); // 1 ...
        cm.recordPrediction(0, 2, cat3); // 1
        cm.recordPrediction(0, 1, cat3); // 1 ...
        cm.recordPrediction(0, 1, cat3); // 1
        cm.recordPrediction(0, 1, cat3); // 2
        cm.recordPrediction(2, 5, cat3); // 1
        cm.recordPrediction(0, 0, cat3); // 1 ...
        cm.recordPrediction(0, 0, cat3); // 2
        cm.recordPrediction(2, 0, cat3); // 1
        cm.recordPrediction(2, 3, cat3); // 2

        baseline = cm.categorySpecificMaxF1Baseline();
        assertEquals(7, (int)baseline.get1());
        assertEquals(20, (int)baseline.get2());
        assertEquals((int) cm.getConfusionMatrix(cat1).majorityNonNilCorrectHits() + cm.getConfusionMatrix(cat3).majorityNonNilCorrectHits(), (int)baseline.get1());
        assertEquals((int) cm.getConfusionMatrix(cat1).getTotal() + cm.getConfusionMatrix(cat3).getTotal(), (int)baseline.get2());

        List<String> reportKeys = new ArrayList<>();
        List<Number> reportValues = new ArrayList<>();
                    
        cm.reportClassSpecificMajorityBaseline("test", new Reporter() {
            @Override
            public void report(String key, Object val) {
                reportKeys.add(key);
                reportValues.add((Number)val);
            }});
        assertEquals(Arrays.asList(
                "testMNNBaselineNumTotal",
                "testMNNBaselineNumPositive",
                "testMNNBaselineNumPossible",
                "testMNNBaselineNumCorrectHits",
//                "testMNNBaselineNumCorrectNils",
//                "testMNNBaselineNumCorrect", 
//                "testMNNBaselineAccuracy",
                "testMNNBaselinePrecision",
                "testMNNBaselineRecall",
                "testMNNBaselineF1"), reportKeys);
        assertArrayEquals(new double[]{
                30, //  "testMNNBaselineNumTotal",
                20, //  "testMNNBaselineNumPositive",
                11, //  "testMNNBaselineNumPossible",
                // 7 correct hits plus 7 correct nils
                7,  //  "testMNNBaselineNumCorrectHits",
//                7,  //  "testMNNBaselineNumCorrectNils",
//                14, //  "testMNNBaselineNumCorrect", 
//                14.0/30, //  "testMNNBaselineAccuracy",
                7.0/20,  //  "testMNNBaselinePrecision",
                7.0/11,  //  "testMNNBaselineRecall",
                2*7.0/20*7.0/11/(7.0/20+7.0/11) // "testMNNBaselineF1"
            },
            TestUtils.toArray(reportValues), 1E-9);
    }
    
}
