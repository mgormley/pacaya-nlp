package edu.jhu.nlp.data;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.TreeMap;

import edu.jhu.nlp.sprl.SprlClassLabel;
import edu.jhu.prim.tuple.Pair;
/**
 * SPRL properties for a particular predicate-argument pair
 *  
 */
public class Properties implements Iterable<Pair<String, Double>> {

    private Map<String, Double> props;

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
        props = new TreeMap<>();
    }

    public int size() {
        return props.size();
    }
    
    public void add(String property, double value) {
        props.put(property, value);
    }

    @Override
    public String toString() {
        StringBuilder argsStr = new StringBuilder();
        argsStr.append("Properties [");
        int i = 0;
        for (Map.Entry<String, Double> p : props.entrySet()) {
            if (i != 0) {
                argsStr.append(", ");
            }
            argsStr.append(p.getKey());
            argsStr.append("=");
            argsStr.append(p.getValue());
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
        return this.props.equals(rhs.props);
    }

	public Double[] toArray() {
		Double a[] = new Double[Property.labels.size()];
		int i = 0;
		for (String k : Property.labels) {
			a[i] = props.get(k);
			i++;
		}
		return a;
	}

    public List<SprlClassLabel> toLabels() {
        List<SprlClassLabel> returnList = new ArrayList<>();
        for (String k : Property.labels) {
            returnList.add(SprlClassLabel.getLabel(props.get(k)));
        }
        return returnList;
    }

    @Override
    public Iterator<Pair<String, Double>> iterator() {
        final Iterator<Entry<String, Double>> itr = props.entrySet().iterator();
        return new Iterator<Pair<String,Double>>() {
            
            @Override
            public Pair<String, Double> next() {
                Entry<String, Double> e = itr.next();
                return new Pair<>(e.getKey(), e.getValue());
            }
            
            @Override
            public boolean hasNext() {
                return itr.hasNext();
            }
        };
    }

	
}
