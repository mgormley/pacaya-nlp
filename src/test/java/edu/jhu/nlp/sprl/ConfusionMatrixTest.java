package edu.jhu.nlp.sprl;

import static org.junit.Assert.assertEquals;

import java.io.StringWriter;
import java.util.Arrays;

import org.junit.Test;

public class ConfusionMatrixTest {

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
}
