package edu.jhu.nlp.sprl;

import static org.junit.Assert.assertEquals;

import java.io.IOException;

import static edu.jhu.nlp.sprl.SprlConcreteEvaluator.hStack;


import org.junit.Test;

public class SprlConcreteEvaluatorTest {

    @Test
    public void testHStack() throws IOException {
        // constructor just to count default
        SprlConcreteEvaluator eval = new SprlConcreteEvaluator();
        String a = String.join("",
                "1234\n",
                "\n",
                "1\n",
                "123\n",
                "1\n"
                );
        String b = String.join("",
                "12\n",
                "123\n",
                "1\n"
                );
        String c = String.join("",
                "1\n",
                "\n",
                "\n",
                "12345\n"
                );
        String d = "";
        String stacked = String.join("",
                "123412 1    \n",
                "    123     \n",
                "1   1       \n",
                "123    12345\n",
                "1           \n"
                );
        assertEquals(stacked, hStack(a,b,c,d)); 
    }
}
