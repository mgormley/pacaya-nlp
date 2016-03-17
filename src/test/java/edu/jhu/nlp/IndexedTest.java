package edu.jhu.nlp;

import static edu.jhu.nlp.Indexed.enumerate;
import static edu.jhu.nlp.Indexed.collect;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.junit.Test; 

public class IndexedTest {

    @Test
    public void testEnumerate() {
        List<String> strings = Arrays.asList("this", "is", "a", "test");
        List<Object> indexesAndVals = new ArrayList<>();
        for (Indexed<String> e : enumerate(strings)) {
            indexesAndVals.add(e.index());
            indexesAndVals.add(e.get());
        }
        List<Object> expected = Arrays.asList(0, "this", 1, "is", 2, "a", 3, "test");
        assertTrue(indexesAndVals.equals(expected));
    }

    @Test
    public void testCollect() {
        List<String> strings = Arrays.asList("this", "is", "a", "test");
        List<Indexed<String>> collected= new ArrayList<>(collect(enumerate(strings)));
        List<Indexed<String>> expected = Arrays.asList(
                new Indexed<String>("this", 0),
                new Indexed<String>("is", 1),
                new Indexed<String>("a", 2),
                new Indexed<String>("test", 3));
        assertTrue(collected.equals(expected));
    }

}
