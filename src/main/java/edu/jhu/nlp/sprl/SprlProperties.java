package edu.jhu.nlp.sprl;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.pacaya.sch.util.DefaultDict;
import edu.jhu.prim.set.IntHashSet;
import edu.jhu.prim.tuple.Pair;
import edu.jhu.prim.tuple.Triple;

/**
 * Basically a map from pred,arg,string triples to string labels;
 *
 */
public class SprlProperties {
    private static final Logger log = LoggerFactory.getLogger(SprlEvaluator.class);
    private SprlLabelConverter labelConverter;
    private IntHashSet preds;
    private Set<Pair<Integer, Integer>> nilPairs;
    private List<Triple<Integer, Integer, String>> labeledProperties;
    private Map<Triple<Integer, Integer, String>, String> propLabels;
    private DefaultDict<Pair<Integer, Integer>, Set<String>> propsByPair;
    private DefaultDict<Integer, Set<Integer>> argsByPred;

    public SprlProperties(SprlProperties sprl) {
        labelConverter = sprl.labelConverter;
        preds = new IntHashSet(sprl.preds);
        nilPairs = new HashSet<>(sprl.nilPairs);
        labeledProperties = new ArrayList<>(sprl.labeledProperties);
        propLabels = new HashMap<>(sprl.propLabels);
        propsByPair = new DefaultDict<>(sprl.propsByPair, v -> new TreeSet<>(v));
        argsByPred = new DefaultDict<>(sprl.argsByPred, v -> new TreeSet<>(v));
    }

    public SprlProperties(SprlLabelConverter labelConverter) {
        this.labelConverter = labelConverter;
        propLabels = new HashMap<>();
        propsByPair = new DefaultDict<>(Void -> new TreeSet<>());
        argsByPred = new DefaultDict<>(Void -> new TreeSet<>());
        nilPairs = new HashSet<>();
        preds = new IntHashSet();
        labeledProperties = new ArrayList<>();
    }

    public void set(int predLoc, int argLoc, String property, String label) {
        Pair<Integer, Integer> pair = new Pair<>(predLoc, argLoc);
        if (SprlLabelConverter.nil().equals(label)) {
            nilPairs.add(pair);
        } else if (nilPairs.contains(pair)) {
            throw new IllegalArgumentException("cannot set a label for a pair that has been declared nil");
        } else {
            Triple<Integer, Integer, String> t = new Triple<>(predLoc, argLoc, property);
            preds.add(predLoc);
            argsByPred.get(predLoc).add(argLoc);
            if (propsByPair.get(pair).add(property)) {
                labeledProperties.add(t);
            } else if (label != get(t)) {
                // TODO: maybe this should throw an exception instead to make sure there isn't corrupt data
                log.warn(String.format("changing sprl %s on (%s, %s) from: %s to %s", property, predLoc, argLoc, get(t), label));  
            }
            propLabels.put(t, label);
        }
    }

    public IntHashSet getPreds() {
        return preds;
    }

    public Set<Pair<Integer, Integer>> getPairs() {
        return propsByPair.keySet();
    }

    public boolean containsPair(Pair<Integer, Integer> e) {
        return propsByPair.containsKey(e);
    }

    public String get(int pred, int arg, String property) {
        return get(new Triple<>(pred, arg, property));
    }

    public Set<Pair<Integer, Integer>> getMarkedNilPairs() {
        return nilPairs;
    }
    
    public Set<String> getLabeledProperties(Pair<Integer, Integer> pair) {
        return propsByPair.getOrDefault(pair, Collections.emptySet());
    }

    public List<Triple<Integer, Integer, String>> getLabeledProperties() {
        return labeledProperties;
    }

    public SprlLabelConverter getConverter() {
        return labelConverter;
    }

    /**
     * if the pred arg pair was unseen or seen as a nil pair, the nil label is
     * returned, otherwise, the property is looked up
     */
    public String get(Triple<Integer, Integer, String> e) {
        String label = propLabels.get(e);
        if (label != null) {
            return label;
        } else if (!propsByPair.containsKey(new Pair<>(e.get1(), e.get2()))) {
            return SprlLabelConverter.nil();
        } else {
            throw new IllegalArgumentException(
                    String.format("The requested property has not been labeled for the given pair: %s", e));
        }
    }

}
