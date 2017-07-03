package edu.jhu.nlp.sprl;

import static edu.jhu.nlp.sprl.ConfusionMatrix.f1;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.jhu.pacaya.sch.util.DefaultDict;
import edu.jhu.pacaya.util.report.Reporter;
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
    private DefaultDict<C, ConfusionMatrix<L>> counts;
    private ConfusionMatrix<L> total;

    public ConfusionMap(Set<L> nilLabels) {
        counts = new DefaultDict<>(Void -> new ConfusionMatrix<>(nilLabels));
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

    /*
     * public int numExamples(L gold, L pred, C category) { return
     * getConfusionMatrix(category).numExamples(gold, pred); }
     */

    public Set<C> getCategories() {
        return counts.keySet();
    }

    public ConfusionMatrix<L> getTotal() {
        return total;
    }

    /**
     * updates the key for the given map m with value if
     * 
     * @param m
     * @param key
     * @param value
     */
    public static void maxIn(Map<Integer, Integer> m, int key, int value) {
        Integer curVal = m.get(key);
        if (curVal == null || value > curVal) {
            m.put(key, value);
        }
    }

    /**
     * Computes the number of correct hits and total predicted that maximizes F1
     * subject to the constraint that a single label must be chosen for each
     * category (the label may be nil)
     * 
     * the only tricky choice is whether to set a label to nil or not; the
     * viterbi algorithm can't naively solve this because F1 doesn't decompose
     * (i.e. I can't separately maximize F1 on the prefix and on the suffix and
     * then stick the solutions together and assume I have the max F1 solution);
     * If I
     * 
     * however, we can do viterbi where we jointly maximize ov we could do it
     * for max recall; instead of just having a score for each choice of whether
     * or not the category i if I can just get the pareto frontier of
     * correcthits and correct misses
     */
    public Pair<Integer, Integer> categorySpecificMaxF1Baseline() {
        // put the categories into a list
        List<C> ordered = new ArrayList<>(getCategories());
        int[] correct = ordered.stream().mapToInt(c -> getConfusionMatrix(c).majorityNonNilCorrectHits()).toArray();
        int[] predicted = ordered.stream().mapToInt(c -> getConfusionMatrix(c).getTotal()).toArray();
        // MaxCorrectForNumPredicted
        HashMap<Integer, Integer> cur = new HashMap<>();
        // initialize
        cur.put(0, 0);
        // loop over classes (i.e. time steps)
        for (int catId = 0; catId < ordered.size(); catId++) {
            HashMap<Integer, Integer> prev = cur;
            cur = new HashMap<>();
            // pick each (correct, predicted) pair and try to extend it in two
            // ways
            for (Map.Entry<Integer, Integer> e : prev.entrySet()) {
                int numPredicted = e.getKey();
                int maxCorrect = e.getValue();

                // predict none
                maxIn(cur, numPredicted, maxCorrect);

                // predict max
                maxIn(cur, numPredicted + predicted[catId], maxCorrect + correct[catId]);
            }
        }
        int possible = getTotal().getNumPossible();

        // map to int pair and max by f1
        return cur.entrySet().stream().map(e -> new Pair<Integer, Integer>(e.getValue(), e.getKey()))
                .max(new Comparator<Pair<Integer, Integer>>() {
                    @Override
                    public int compare(Pair<Integer, Integer> o1, Pair<Integer, Integer> o2) {
                        return Double.compare(f1(o1.get1(), o1.get2(), possible), f1(o2.get1(), o2.get2(), possible));
                    }
                }).get();

    }

    public ConfusionMatrix<L> getConfusionMatrix(C category) {
        return counts.get(category);
    }

    // TODO: follow back pointers to get the actual opt F1 tagging and
    // thus the number of correct nils, and thus the total correct
    // and the accuracy
    public void reportClassSpecificMajorityBaseline(String name, Reporter rep) {
        // majority non-nil baseline if we are allow to choose a different
        Pair<Integer, Integer> correctAndPositive = categorySpecificMaxF1Baseline();
        int correctHits = correctAndPositive.get1();
        // int correctNils = getTotal().getCorrectNils();
        int positive = correctAndPositive.get2();
        int total = getTotal().getTotal();
        int possible = getTotal().getNumPossible();
        // int correct = correctHits + correctNils;
        rep.report(name + "MNNBaselineNumTotal", total);
        rep.report(name + "MNNBaselineNumPositive", positive);
        rep.report(name + "MNNBaselineNumPossible", possible);
        rep.report(name + "MNNBaselineNumCorrectHits", correctHits);
        // rep.report(name + "MNNBaselineNumCorrectNils", correctNils);
        // rep.report(name + "MNNBaselineNumCorrect", correct);
        // rep.report(name + "MNNBaselineAccuracy",
        // ConfusionMatrix.accuracy(correct, total));
        rep.report(name + "MNNBaselinePrecision", ConfusionMatrix.precision(correctHits, positive));
        rep.report(name + "MNNBaselineRecall", ConfusionMatrix.recall(correctHits, possible));
        rep.report(name + "MNNBaselineF1", ConfusionMatrix.f1(correctHits, positive, possible));
    }

    /*
     * public void print(Collection<L> labelOrder, Writer out) throws
     * IOException { total.print("totalmicro", labelOrder, out); for
     * (Map.Entry<C, ConfusionMatrix<L>> e : counts.entrySet()) { String
     * category = e.getKey().toString(); e.getValue().print(category,
     * labelOrder, out); } out.write("\n"); }
     */
}
