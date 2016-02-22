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
        getConfusionMatrix(category).recordPrediction(gold, pred, example);
        total.recordPrediction(gold, pred);
    }

    public boolean hasExample(L gold, L pred, C category) {
        return getConfusionMatrix(category).hasExamples(gold, pred);
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
        for (Map.Entry<C, ConfusionMatrix<L>> e : counts.entrySet()) {
            String category = e.getKey().toString();
            out.write(e.getValue().format(category, labelOrder));
            // write out an example for each non-zero cell in the matrix
            for (Map.Entry<Pair<L, L>, List<String>> exList : e.getValue().getExamples().entrySet()) {
                for (String ex : exList.getValue()) {
                    out.write(String.format("%s %s %s:\n%s\n", category, exList.getKey().get1().toString(),
                            exList.getKey().get2().toString(), ex));
                }
            }
        }
        out.write("\n");
    }

}
