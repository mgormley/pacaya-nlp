package edu.jhu.nlp.sprl;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.TreeSet;

import org.junit.Test;

import edu.jhu.pacaya.sch.util.TestUtils;
import edu.jhu.prim.tuple.Pair;
import edu.jhu.prim.tuple.Triple;

public class SprlPropertiesTest {

    private void checkProperties1(SprlProperties props, BinarySprlLabelConverter converter) {
        // nils shouldn't show up the same as those labeled with non-nils
        // also, after marking a pair as nil, you can't mark it as something else 
        assertTrue(TestUtils.checkThrows(() -> props.set(0, 1, "awareness", SprlLabelConverter.UNLIKELY),
                IllegalArgumentException.class));
        assertTrue(TestUtils.checkThrows(() -> props.set(0, 1, "volitional", SprlLabelConverter.UNLIKELY),
                IllegalArgumentException.class));
        assertEquals(new TreeSet<String>(Arrays.asList()), props.getLabeledProperties(new Pair<>(0, 1)));

        props.set(3, 1, "volitional", SprlLabelConverter.nil());
        assertEquals(new HashSet<Pair<Integer, Integer>>(Arrays.asList(new Pair<>(0, 1), new Pair<>(3, 1))),
                props.getMarkedNilPairs());
        assertTrue(TestUtils.checkThrows(() -> props.set(0, 1, "awareness", SprlLabelConverter.UNLIKELY),
                IllegalArgumentException.class));
        assertEquals(SprlLabelConverter.LIKELY, props.get(new Triple<>(4, 3, "otherthing")));
        assertEquals(SprlLabelConverter.UNLIKELY, props.get(new Triple<>(1, 0, "volitional")));
        assertEquals(SprlLabelConverter.LIKELY, props.get(new Triple<>(1, 0, "awareness")));
        assertTrue(props.getConverter() == converter);
        assertEquals(SprlLabelConverter.LIKELY, props.get(4, 3, "otherthing"));
        assertEquals(SprlLabelConverter.UNLIKELY, props.get(1, 0, "volitional"));
        assertEquals(SprlLabelConverter.LIKELY, props.get(1, 0, "awareness"));
        assertEquals(SprlLabelConverter.NOT_AN_ARG, props.get(0, 1, "volitional"));
        assertEquals(SprlLabelConverter.NOT_AN_ARG, props.get(0, 1, "awareness"));
        assertEquals(SprlLabelConverter.NOT_AN_ARG, props.get(0, 1, "otherthing"));
        assertEquals(SprlLabelConverter.NOT_AN_ARG, props.get(0, 1, "evenotherthing"));
        assertEquals(SprlLabelConverter.NOT_AN_ARG, props.get(5, 4, "awareness"));
        // can't ask for property on labeled edge that hasn't been labeled
        assertTrue(TestUtils.checkThrows(() -> props.get(1, 0, "otherthing"), IllegalArgumentException.class));
        assertEquals(Arrays.asList(new Triple<>(1, 0, "awareness"), new Triple<>(1, 0, "volitional"),
                new Triple<>(1, 2, "awareness"), new Triple<>(1, 2, "volitional"), new Triple<>(4, 3, "otherthing")),
                props.getLabeledProperties());
        assertEquals(new TreeSet<String>(Arrays.asList("awareness", "volitional")),
                props.getLabeledProperties(new Pair<>(1, 0)));
        assertEquals(new TreeSet<String>(Arrays.asList("awareness", "volitional")),
                props.getLabeledProperties(new Pair<>(1, 2)));
        assertEquals(new TreeSet<String>(Arrays.asList("otherthing")), props.getLabeledProperties(new Pair<>(4, 3)));
        assertEquals(new HashSet<Pair<Integer, Integer>>(Arrays.asList(
                new Pair<>(1, 0),
                new Pair<>(1, 2),
                new Pair<>(4, 3))),
                props.getPairs());
        assertTrue(props.containsPair(new Pair<>(1, 0)));
        assertTrue(props.containsPair(new Pair<>(1, 2)));
        assertTrue(props.containsPair(new Pair<>(4, 3)));
        assertFalse(props.containsPair(new Pair<>(0, 1)));
        assertFalse(props.containsPair(new Pair<>(5, 4)));

        // check preds
        ArrayList<Integer> preds = new ArrayList<>();
        for (int i : props.getPreds().toNativeArray()) preds.add(i);
        preds.sort(null);
        assertEquals(Arrays.asList(1, 4), preds); 
    }
    
    @Test
    public void test() {
        BinarySprlLabelConverter converter = new BinarySprlLabelConverter(3.5);
        SprlProperties props = new SprlProperties(converter);
        props.set(1, 0, "awareness", SprlLabelConverter.LIKELY);
        props.set(1, 0, "volitional", SprlLabelConverter.UNLIKELY);
        props.set(1, 2, "awareness", SprlLabelConverter.UNLIKELY);
        props.set(1, 2, "volitional", SprlLabelConverter.UNLIKELY);
        props.set(4, 3, "otherthing", SprlLabelConverter.LIKELY);
        props.set(0, 1, "volitional", SprlLabelConverter.nil());
        checkProperties1(props, converter);
        // make a copy
        SprlProperties newProps = new SprlProperties(props);
        checkProperties1(newProps, converter);
        // make sure the original is still the same
        checkProperties1(props, converter);
        // add to the new props
        newProps.set(1, 2, "otherthing", SprlLabelConverter.LIKELY);
        newProps.set(5, 4, "otherthing", SprlLabelConverter.UNLIKELY);
        // make sure the change was made
        assertEquals(SprlLabelConverter.LIKELY, newProps.get(1, 2, "otherthing"));
        // make sure the old didn't change
        checkProperties1(props, converter);

//      // (before, couldn't set to different value
//      assertTrue(TestUtils.checkThrows(() -> props.set(1, 0, "awareness", SprlLabelConverter.UNLIKELY),
//              IllegalArgumentException.class));
        props.set(1, 0, "awareness", SprlLabelConverter.UNLIKELY);
        assertEquals(SprlLabelConverter.UNLIKELY, props.get(1, 0, "awareness"));
        // change it again (no warning; which I'm not actually checking)
        props.set(1, 0, "awareness", SprlLabelConverter.UNLIKELY);
        assertEquals(SprlLabelConverter.UNLIKELY, props.get(1, 0, "awareness"));
    }

}
