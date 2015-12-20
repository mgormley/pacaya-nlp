package edu.jhu.nlp.data;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import edu.jhu.prim.tuple.Pair;
/**
 * SPRL properties for a particular predicate-argument pair
 *  
 */
public class Properties implements Iterable<Pair<String, Double>> {

    private List<Pair<String, Double>> props;
    public static enum Property {
        awareness,
        change_of_location,
        change_of_state,
        changes_possession,
        created,
        destroyed,
        existed_after,
        existed_before,
        existed_during,
        exists_as_physical,
        instigation,
        location_of_event,
        makes_physical_contact,
        manipulated_by_another,
        predicate_changed_argument,
        sentient,
        stationary,
        volition;
        
        // add the labels as string names
        public static ArrayList<String> labels;
        static {
            labels = new ArrayList<>();
            for (Property label : values()) {
                labels.add(label.name());
            }
        }
    };

    public static final int nquestions = Property.labels.size();    

    public Properties() {
        props = new ArrayList<>();
    }

    public Properties(Properties other) {
        props = new ArrayList<>(other.props);
    }

    public List<Pair<String, Double>> get() {
        return props;
    }

    /**
       Add the properties to a map and default values for anything in
       'required' not in properties
     */
    public Map<String, Double> toMap(Iterable<String> required, Double defaultValue) {
        // add everything in properties to the map
        Map<String, Double> returnMap = new TreeMap<>();
        for (Pair<String, Double> prop : props) {
            returnMap.put(prop.get1(), prop.get2());
        }
        if (required != null) {
            for (String req : required) {
                if (!returnMap.containsKey(req)) {
                    returnMap.put(req, defaultValue);
                }
            }
        }
        return returnMap;
    }

    public Map<String, Double> toMap() {
        return toMap(null, null);
    }

    public Iterator<Pair<String, Double>> iterator() {
        return props.iterator();
    }

    public void add(String property, double value) {
        props.add(new Pair<>(property, value));
    }

    @Override
    public String toString() {
        StringBuilder argsStr = new StringBuilder();
        argsStr.append("Properties [");
        int i = 0;
        for (Pair<String, Double> p : props) {
            if (i != 0) {
                argsStr.append(", ");
            }
            argsStr.append(p.get1());
            argsStr.append("=");
            argsStr.append(p.get2());
            i++;
        }
        argsStr.append("]");
        return argsStr.toString();
        
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null) return false;
        if (getClass() != obj.getClass()) return false;
        Properties rhs = (Properties) obj;

        // check the lengths
        if (props.size() != rhs.props.size()) return false;

        // check each item
        Iterator<Pair<String, Double>> lhsItr = props.iterator();
        Iterator<Pair<String, Double>> rhsItr = rhs.props.iterator();
        while (lhsItr.hasNext()) {
            if (lhsItr.next().get1() != rhsItr.next().get1()) {
                return false;
            }
        }
        return true;
    }

	public double[] toArray() {
		double a[] = new double[Property.labels.size()];
		Map<String, Double> m = toMap(Property.labels, -1.0);
		int i = 0;
		for (String k : Property.labels) {
			assert m.containsKey(k);
			a[i] = m.get(k);
			i++;
		}
		return a;
	}

	
}
