package edu.jhu.nlp.sprl;

import java.io.IOException;
import java.io.Writer;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.jhu.prim.tuple.Pair;

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
    private HashMap<C, ConfusionMatrix<L>> counts;
    private ConfusionMatrix<L> total;
    private Set<L> nilLabels;

    public ConfusionMap(Set<L> nilLabels) {
        counts = new HashMap<>();
        this.nilLabels = nilLabels;
        total = new ConfusionMatrix<>(nilLabels);
    }

    public void recordPrediction(L gold, L pred, C category) {
        recordPrediction(gold, pred, category, null);
    }

    public void recordPrediction(L gold, L pred, C category, String example) {
        recordPrediction(gold, pred, category, example, Integer.MAX_VALUE);
    }
    
    public void recordPrediction(L gold, L pred, C category, String example, int maxNumExamples) {
        getConfusionMatrix(category).recordPrediction(gold, pred, example, maxNumExamples);
        total.recordPrediction(gold, pred, example, maxNumExamples);
    }

    public int numExamples(L gold, L pred, C category) {
        return getConfusionMatrix(category).numExamples(gold, pred);
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
        total.print("total", labelOrder, out);
        for (Map.Entry<C, ConfusionMatrix<L>> e : counts.entrySet()) {
            String category = e.getKey().toString();
            e.getValue().print(category, labelOrder, out);
        }
        out.write("\n");
    }

}
