package edu.jhu.nlp.sprl;

import java.util.HashMap;
import java.util.Set;

/**
 * Holds integer counts for things of type T; default count is 0 and an entry is
 * only created when add is called
 *
 * @param <T>
 *            Type of things being counted
 */
public class Counter<T> {
    private HashMap<T, Integer> counts;
    private int total;

    public Counter() {
        counts = new HashMap<>();
        total = 0;
    }

    public void add(T obj) {
        add(obj, 1);
    }

    public void add(T obj, int incrementBy) {
        counts.put(obj, get(obj) + incrementBy);
        total += incrementBy;
    }

    public Set<T> keySet() {
        return counts.keySet();
    }

    public int get(T obj) {
        Integer count = counts.get(obj);
        if (count == null) {
            count = 0;
        }
        return count;
    }

    public int getTotal() {
        return total;
    }
}
