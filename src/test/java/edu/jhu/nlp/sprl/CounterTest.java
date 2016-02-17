package edu.jhu.nlp.sprl;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Collections;

import org.junit.Test;

public class CounterTest {

    @Test
    public void testCounts() {
        Counter<Integer> c = new Counter<>();
        assertEquals(c.keySet().size(), 0);
        assertEquals(c.get(9), 0);
        assertEquals(c.getTotal(), 0);
        c.add(9);
        assertEquals(c.keySet(), Collections.singleton(9));
        assertEquals(c.getTotal(), 1);
        assertEquals(c.get(9), 1);
        c.add(8);
        c.add(9);
        assertEquals(c.get(9), 2);
        assertEquals(c.get(8), 1);
        assertEquals(c.getTotal(), 3);
        c.add(8, 5);
        c.add(9, 10);
        assertEquals(c.get(8), 6);
        assertEquals(c.get(9), 12);
        assertEquals(c.getTotal(), 18);
        c.add(8, -20);
        c.add(9, -30);
        assertEquals(c.get(8), -14);
        assertEquals(c.get(9), -18);
        assertEquals(c.getTotal(), -32);
        assertEquals(c.keySet().size(), 2);
        assertTrue(c.keySet().contains(8));
        assertTrue(c.keySet().contains(9));
    }
}
