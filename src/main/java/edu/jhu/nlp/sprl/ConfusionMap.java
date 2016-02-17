package edu.jhu.nlp.sprl;

import java.util.HashMap;
import java.util.Set;

/**
 * A map of confusion matrixes
 * 
 * @param <L>
 *            (label) The kind of things being predicted
 * @param <C>
 *            (category) The kind of label for the different confusion matrixes
 */
public class ConfusionMap<L, C> {
    /**
     * counters for each category
     */
    HashMap<C, ConfusionMatrix<L>> counts;
    ConfusionMatrix<L> total;
    L nilValue;

    public ConfusionMap(L inNilValue) {
        counts = new HashMap<>();
        total = new ConfusionMatrix<>();
        nilValue = inNilValue;
    }

    public void recordPrediction(L gold, L pred, C category) {
        getConfusionMatrix(category).recordPrediction(gold, pred);
        total.recordPrediction(gold, pred);
    }

    public Set<C> getCategories() {
        return counts.keySet();
    }

    public ConfusionMatrix<L> getTotal() {
        return total;
    }
    
    public ConfusionMatrix<L> getConfusionMatrix(C category) {
        ConfusionMatrix<L> cm = counts.get(category);
        if (cm == null) {
            cm = new ConfusionMatrix<>();
            counts.put(category, cm);
        }
        return cm;
    }

}
