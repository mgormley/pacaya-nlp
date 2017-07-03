package edu.jhu.nlp.sprl;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import java.io.IOException;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;

import org.junit.Test;

import edu.jhu.pacaya.sch.util.TestUtils;
import edu.jhu.pacaya.util.report.Reporter;

public class ConfusionMatrixTest {
    private static double tol = 1E-9;
    
    @Test
    public void testCounts() {
        ConfusionMatrix<Integer> cm = new ConfusionMatrix<>();
        cm.recordPrediction(3, 3); // 1 ...
        cm.recordPrediction(0, 2); // 1
        cm.recordPrediction(0, 1); // 1 ...
        cm.recordPrediction(3, 1); // 1
        cm.recordPrediction(0, 1); // 2
        cm.recordPrediction(3, 5); // 1
        cm.recordPrediction(0, 0); // 1 ...
        cm.recordPrediction(0, 0); // 2
        cm.recordPrediction(2, 0); // 1
        cm.recordPrediction(3, 3); // 2
        assertEquals(cm.getCount(0, 0), 2);
        assertEquals(cm.getCount(0, 1), 2);
        assertEquals(cm.getCount(0, 2), 1);
        assertEquals(cm.getCount(2, 0), 1);
        assertEquals(cm.getCount(2, 2), 0);
        assertEquals(cm.getCount(3, 1), 1);
        assertEquals(cm.getCount(3, 3), 2);
        assertEquals(cm.getCount(3, 5), 1);
        assertEquals(cm.getCount(4, 4), 0);
        assertEquals(cm.getTotal(), 10);
        assertEquals(cm.getGoldCount(0), 5);
        assertEquals(cm.getGoldCount(1), 0);
        assertEquals(cm.getGoldCount(2), 1);
        assertEquals(cm.getGoldCount(3), 4);
        assertEquals(cm.getGoldCount(4), 0);
        assertEquals(cm.getGoldCount(5), 0);
        assertEquals(cm.getPredCount(0), 3);
        assertEquals(cm.getPredCount(1), 3);
        assertEquals(cm.getPredCount(2), 1);
        assertEquals(cm.getPredCount(3), 2);
        assertEquals(cm.getPredCount(5), 1);
        assertEquals(cm.getPredCount(6), 0);
        assertEquals(cm.getCorrect(), 4);
        StringWriter sw = new StringWriter();
        sw.write("gold \\ pred  &  0  &  1  &  2  &  3  &  5  &  total  \\\\\n");
        sw.write("          0  &  2  &  2  &  1  &  0  &  0  &      5  \\\\\n");
        sw.write("          1  &  0  &  0  &  0  &  0  &  0  &      0  \\\\\n");
        sw.write("          2  &  1  &  0  &  0  &  0  &  0  &      1  \\\\\n");
        sw.write("          3  &  0  &  1  &  0  &  2  &  1  &      4  \\\\\n");
        sw.write("          5  &  0  &  0  &  0  &  0  &  0  &      0  \\\\\n");
        sw.write("      total  &  3  &  3  &  1  &  2  &  1  &     10  \\\\\n");
        assertEquals(sw.toString(), cm.formatMatrix(Arrays.asList(0, 1, 2, 3, 5)));
    }

    @Test
    public void testNilCounts() throws IOException {
        ConfusionMatrix<Integer> cm = new ConfusionMatrix<>(Collections.singleton(0));
        assertEquals(Collections.emptySet(), cm.keySet());
        assertEquals(1.0, cm.recall(), tol);
        assertEquals(1.0, cm.precision(), tol);
        assertEquals(1.0, cm.f1(), tol);
        assertEquals(1.0, cm.accuracy(), tol);
        assertEquals(0, cm.majorityNonNilCorrectHits());
        assertEquals(1.0, cm.majorityNonNilPrecision(), tol);
        assertEquals(1.0, cm.majorityNonNilRecall(), tol);
        assertEquals(1.0, cm.majorityNonNilF1(), tol);
        cm.recordPrediction(0, 2); // 1
        assertEquals(new HashSet<>(Arrays.asList(0, 2)), cm.keySet());
        assertEquals(1.0, cm.recall(), tol);
        assertEquals(0.0, cm.precision(), tol);
        assertEquals(0.0, cm.f1(), tol);
        assertEquals(0.0, cm.accuracy(), tol);
        cm.recordPrediction(3, 1); // 1
        assertEquals(0.0, cm.recall(), tol);
        assertEquals(0.0, cm.precision(), tol);
        assertEquals(0.0, cm.f1(), tol);
        assertEquals(0.0, cm.accuracy(), tol);
        cm.recordPrediction(3, 3, "example 1"); // 1 ...
        cm.recordPrediction(0, 1); // 1 ...
        cm.recordPrediction(0, 1); // 2
        cm.recordPrediction(3, 5); // 1
        cm.recordPrediction(0, 0); // 1 ...
        cm.recordPrediction(0, 0); // 2
        cm.recordPrediction(2, 0); // 1
        cm.recordPrediction(3, 3, "example 2", 1); // 2
        assertEquals(cm.getCount(0, 0), 2);
        assertEquals(cm.getCount(0, 1), 2);
        assertEquals(cm.getCount(0, 2), 1);
        assertEquals(cm.getCount(2, 0), 1);
        assertEquals(cm.getCount(2, 2), 0);
        assertEquals(cm.getCount(3, 1), 1);
        assertEquals(cm.getCount(3, 3), 2);
        assertEquals(cm.getCount(3, 5), 1);
        assertEquals(cm.getCount(4, 4), 0);
        assertEquals(cm.getTotal(), 10);
        assertEquals(cm.getExpectedNils(), 5);
        assertEquals(cm.getPredictedNils(), 3);
        assertEquals(cm.getCorrectNils(), 2);
        assertEquals(cm.getCorrectHits(), 2);
        assertEquals(cm.getNumPossible(), 5);
        assertEquals(cm.getNumPositive(), 7);
        assertEquals(2.0 / 5.0, cm.recall(), tol);
        assertEquals(2.0 / 7.0, cm.precision(), tol);
        assertEquals(8.0 / 35.0 / (2.0 / 5.0 + 2.0 / 7.0), cm.f1(), tol);
        assertEquals((Integer) 3, cm.majorityNonNilLabel());
        assertEquals(4, cm.majorityNonNilCorrectHits());
        assertEquals(0.8, cm.majorityNonNilRecall(), tol);
        assertEquals(0.4, cm.majorityNonNilPrecision(), tol);
        assertEquals(2*0.8*0.4/1.2, cm.majorityNonNilF1(), tol);
        assertEquals(2*0.8*0.4/1.2, ConfusionMatrix.harmonicMean(0.8,  0.4), tol);
        assertEquals(0, cm.numExamples(0, 0));
        assertEquals(1, cm.numExamples(3, 3));
        assertEquals(Arrays.asList("example 2"), cm.getExamples(3, 3));
        assertEquals(1, cm.getExamples().size());
        assertEquals(cm.getGoldCount(0), 5);
        assertEquals(cm.getGoldCount(1), 0);
        assertEquals(cm.getGoldCount(2), 1);
        assertEquals(cm.getGoldCount(3), 4);
        assertEquals(cm.getGoldCount(4), 0);
        assertEquals(cm.getGoldCount(5), 0);
        assertEquals(cm.getPredCount(0), 3);
        assertEquals(cm.getPredCount(1), 3);
        assertEquals(cm.getPredCount(2), 1);
        assertEquals(cm.getPredCount(3), 2);
        assertEquals(cm.getPredCount(5), 1);
        assertEquals(cm.getPredCount(6), 0);
        assertEquals(cm.getCorrect(), 4);        

        StringWriter sw = new StringWriter();
        sw.write("\n   \tname\tf1(prec)\tmajf1(majprec)\tcorhits\tmajcorhits\tpsbhits\ttotal\n");
        sw.write("~~~\ttesting\t33.3 (28.6)\t53.3 (40.0)\t2\t4\t5\t10\n");
        sw.write("==testing Precision: 0.2857142857142857\n");
        sw.write("==testing Recall: 0.4\n");
        sw.write("==testing F1: 0.3333333333333333\n");
        sw.write("==testing Accuracy: 0.4\n");
        sw.write("==testing MarjoirtyNonNilBaseline: 0.5333333333333333\n");
        sw.write("gold \\ pred  &  0  &  1  &  2  &  3  &  5  &  total  \\\\\n");
        sw.write("          0  &  2  &  2  &  1  &  0  &  0  &      5  \\\\\n");
        sw.write("          1  &  0  &  0  &  0  &  0  &  0  &      0  \\\\\n");
        sw.write("          2  &  1  &  0  &  0  &  0  &  0  &      1  \\\\\n");
        sw.write("          3  &  0  &  1  &  0  &  2  &  1  &      4  \\\\\n");
        sw.write("          5  &  0  &  0  &  0  &  0  &  0  &      0  \\\\\n");
        sw.write("      total  &  3  &  3  &  1  &  2  &  1  &     10  \\\\\n");
        sw.write("testing 3 3:\n");
        sw.write("example 2\n");
        StringWriter ow = new StringWriter();
        List<Integer> labelOrder = Arrays.asList(0, 1, 2, 3, 5);
        cm.print("testing", labelOrder, ow);
        assertEquals(sw.toString(), ow.toString());

        {
            List<String> reportKeys = new ArrayList<>();
            List<Number> reportValues = new ArrayList<>();
            
            cm.reportMajorityBaseline("test", new Reporter() {
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
                    "testMNNBaselineNumCorrectNils",
                    "testMNNBaselineNumCorrect",
                    "testMNNBaselineAccuracy",
                    "testMNNBaselinePrecision",
                    "testMNNBaselineRecall",
                    "testMNNBaselineF1"), reportKeys);
            assertArrayEquals(new double[]{
                    10, // "testMNNBaselineNumTotal",
                    10, // "testMNNBaselineNumPositive",
                    5,  // "testMNNBaselineNumPossible",
                    4,  // "testMNNBaselineNumCorrectHits",
                    0,  // "testMNNBaselineNumCorrectNils",
                    4,  // "testMNNBaselineNumCorrect",
                    0.4, // "testMNNBaselineAccuracy",
                    0.4, // "testMNNBaselinePrecision",
                    0.8, // "testMNNBaselineRecall",
                    2*0.4*0.8/(0.4+0.8) // "testMNNBaselineF1"));
            }, TestUtils.toArray(reportValues), 1E-9);
            
        }
        {
            ArrayList<String> reportKeys = new ArrayList<>();
            ArrayList<Number> reportValues = new ArrayList<>();
            
            cm.reportSummary("test", new Reporter() {
                @Override
                public void report(String key, Object val) {
                    reportKeys.add(key);
                    reportValues.add((Number)val);
                }});
            assertEquals(Arrays.asList(
                    "testNumTotal",
                    "testNumPositive",
                    "testNumPossible",
                    "testNumCorrectHits",
                    "testNumCorrectNils",
                    "testNumCorrect",
                    "testAccuracy",
                    "testPrecision",
                    "testRecall",
                    "testF1"), reportKeys);             
            assertEquals(cm.getTotal(), 10);
            assertEquals(cm.getExpectedNils(), 5);
            assertEquals(cm.getPredictedNils(), 3);
            assertEquals(cm.getCorrectNils(), 2);
            assertEquals(cm.getCorrectHits(), 2);
            assertEquals(cm.getNumPossible(), 5);
            assertEquals(cm.getNumPositive(), 7);
            assertEquals(2.0 / 5.0, cm.recall(), tol);
            assertEquals(2.0 / 7.0, cm.precision(), tol);
            assertEquals(8.0 / 35.0 / (2.0 / 5.0 + 2.0 / 7.0), cm.f1(), tol);
            
            assertArrayEquals(new double[]{
                    10, // "testNumTotal",         
                    7, // "testNumPositive",      
                    5,  // "testNumPossible",      
                    2,  // "testNumCorrectHits",   
                    2,  // "testNumCorrectNils",   
                    4,  // "testNumCorrect",       
                    0.4, //"testAccuracy",         
                    2.0/7, //"testPrecision",        
                    2.0/5, //"testRecall",           
                    2*2.0/7*2.0/5/(2.0/7+2.0/5), // "testF1"));             
            }, TestUtils.toArray(reportValues), 1E-9);
            
        }
        
        cm.recordPrediction(3, 3, "example 3", 2); // 3
        cm.recordPrediction(3, 3, "example 4", 2); // 4
        assertEquals(Arrays.asList("example 4", "example 3"), cm.getExamples(3, 3));
        assertEquals(2, cm.numExamples(3, 3));
        assertEquals(1.0, ConfusionMatrix.accuracy(0,  0), tol);
    }
    
    
//    @Test
//    public void testNilCounts() {
//        ConfusionMatrix<Integer> cm = new ConfusionMatrix<>();
//        cm.recordPrediction(3, 3); // 1 ...
//        cm.recordPrediction(0, 2); // 1
//        cm.recordPrediction(0, 1); // 1 ...
//        cm.recordPrediction(3, 1); // 1
//        cm.recordPrediction(0, 1); // 2
//        cm.recordPrediction(3, 5); // 1
//        cm.recordPrediction(0, 0); // 1 ...
//        cm.recordPrediction(0, 0); // 2
//        cm.recordPrediction(2, 0); // 1
//        cm.recordPrediction(3, 3); // 2
//
//    }

}
