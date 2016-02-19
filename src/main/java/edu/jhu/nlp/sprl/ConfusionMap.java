package edu.jhu.nlp.sprl;

import java.io.IOException;
import java.io.Writer;
import java.util.Collection;
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
    Set<L> nilLabels;

    public ConfusionMap(Set<L> nilLabels) {
        counts = new HashMap<>();
        this.nilLabels = nilLabels;
        total = new ConfusionMatrix<>(nilLabels);
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
            cm = new ConfusionMatrix<>(nilLabels);
            counts.put(category, cm);
        }
        return cm;
    }

    public void print(Collection<L> labelOrder, Writer out) throws IOException {
        out.write(total.format("total", labelOrder));
        for (C k : counts.keySet()) {
            out.write(counts.get(k).format(k.toString(), labelOrder));
        }
        out.write("\n");
    }
    
}
