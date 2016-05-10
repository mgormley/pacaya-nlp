package edu.jhu.nlp.sprl;

import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.jhu.prim.set.IntHashSet;
import edu.jhu.prim.tuple.Pair;
import edu.jhu.prim.tuple.Triple;

/**
 * Basically a map from pred,arg,string triples to string labels;
 *
 */
public class SprlProperties {
    // TODO: switch to defaultdict
    //private DefaultDict<Pair<Integer, Integer>, List<String>>;
    private Map<Pair<Integer, Integer>, Set<String>> propsByPair;
//    private DefaultDict<Integer, List<Integer>>
    private Map<Triple<Integer, Integer, String>, String> propLabels; 
    private Set<Pair<Integer, Integer>> nilPairs;
    private SprlLabelConverter labelConverter;
    
    public SprlProperties(SprlProperties sprl) {
        // TODO Auto-generated constructor stub
    }

    public SprlProperties(SprlLabelConverter labelConverter) {
        this.labelConverter = labelConverter;
    }

    public void set(int predLoc, int argLoc, String value, String label) {
        Pair<Integer, Integer> pair = new Pair<>(predLoc, argLoc);
        if (SprlLabelConverter.nil().equals(label)) {
            nilPairs.add(pair);
        } else if (nilPairs.contains(pair)) {
            throw new IllegalArgumentException("cannot set a label for a pair that has been declared nil");
        } else {
            
        }
    }

    
    public IntHashSet getPreds() {
        // TODO Auto-generated method stub
        return null;
    }

    public Set<Pair<Integer, Integer>> getKnownPairs() {
        // TODO Auto-generated method stub
        return null;
    }

    public boolean containsPair(Pair<Integer, Integer> e) {
        // TODO Auto-generated method stub
        return false;
    }

    /**
     * if the pred arg pair was unseen or seen as a nil pair, the nil label is returned, otherwise, the property is looked up
     */
    public String get(int pred, int arg, String property) {
        return get(new Triple<>(pred, arg, property));
    }

    public Set<String> getLabeledProperties(Pair<Integer, Integer> pair) {
        // TODO Auto-generated method stub
        return null;
    }

    public List<Triple<Integer, Integer, String>> getLabeledProperties() {
        // TODO Auto-generated method stub
        return null;
    }
    
    public SprlLabelConverter getConverter() {
        return labelConverter;
    }

    public String get(Triple<Integer, Integer, String> e) {
        String label = propLabels.get(e);
        if (label != null) {
            return label;
        } else if (!propsByPair.containsKey(new Pair<>(e.get1(), e.get2()))) {
            return SprlLabelConverter.nil();
        } else {
            throw new IllegalArgumentException(String.format("The requested property has not been labeled for the given pair: %s", e));
        }
    }

    
}
