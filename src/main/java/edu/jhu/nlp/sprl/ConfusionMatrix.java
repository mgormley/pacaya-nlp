package edu.jhu.nlp.sprl;

import java.io.IOException;
import java.io.StringWriter;
import java.io.Writer;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.jhu.prim.tuple.Pair;

public class ConfusionMatrix<L> {

    /**
     * labeled confusion matrix
     */
    private Counter<Pair<L, L>> goldPredPairCounts;

    /**
     * optional example for non-zero cells
     */
    private HashMap<Pair<L, L>, List<String>> examples;

    private Counter<L> goldCounts;
    private Counter<L> predCounts;
    private Set<L> keys;
    private Set<L> nilLabels;

    /**
     * unlabeled confusion matrix (coarsen labels to binary based on whether or
     * not they match the nilLabel)
     */
    // Counter<Pair<Boolean, Boolean>> predGoldHitPaircounts;

    public ConfusionMatrix() {
        this(null);
    }

    public ConfusionMatrix(Set<L> nilLabels) {
        this.nilLabels = nilLabels;
        goldPredPairCounts = new Counter<>();
        goldCounts = new Counter<>();
        predCounts = new Counter<>();
        examples = new HashMap<>();
        keys = new HashSet<>();
    }

    /**
     * record the prediction without an example
     */
    public void recordPrediction(L gold, L pred) {
        recordPrediction(gold, pred, null, Integer.MAX_VALUE);
    }

    public void recordPrediction(L gold, L pred, String example) {
        recordPrediction(gold, pred, example, Integer.MAX_VALUE);
    }

    // TODO: allow to provide a number of examples so that the longest example is evicted if we have too many
    public void recordPrediction(L gold, L pred, String example, int numExamplesCap) {
        // count the pair
        goldPredPairCounts.add(new Pair<>(gold, pred));
        goldCounts.add(gold);
        predCounts.add(pred);
        // keep track of keys
        keys.add(gold);
        keys.add(pred);

        // record the example if there is one
        if (example != null) {
            List<String> exampleList = examples.get(new Pair<>(gold, pred));
            if (exampleList == null) {
                exampleList = new ArrayList<>();
                examples.put(new Pair<>(gold, pred), exampleList);
            }
            if (exampleList.size() > 0 && exampleList.size() >= numExamplesCap) {
                int ixOfMaxLength = 0;
                int maxLength = 0;
                for (int i = 0; i < exampleList.size(); i++) {
                    int exLen = exampleList.get(i).length();
                    if (exLen > maxLength) {
                        maxLength = exLen;
                        ixOfMaxLength = i;
                    }
                }
                exampleList.set(ixOfMaxLength, example);
            } else {
                exampleList.add(example);
            }
        }
    }

    public int getCorrect() {
        int total = 0;
        for (L k : goldCounts.keySet()) {
            total += getCount(k, k);
        }
        return total;
    }

    public int getCorrectNils() {
        int total = 0;
        for (L k : nilLabels) {
            total += getCount(k, k);
        }
        return total;
    }

    public int getExpectedNils() {
        int total = 0;
        for (L k : nilLabels) {
            total += getGoldCount(k);
        }
        return total;
    }

    public int getPredictedNils() {
        int total = 0;
        for (L k : nilLabels) {
            total += getPredCount(k);
        }
        return total;
    }

    public int getCorrectHits() {
        return getCorrect() - getCorrectNils();
    }

    public int getPredictedHits() {
        return getTotal() - getPredictedNils();
    }

    public int getPossibleHits() {
        return getTotal() - getExpectedNils();
    }
    
    public double recall() {
        int possible = getPossibleHits();
        if (possible == 0) {
            return 1.0;
        } else {
            return ((double) getCorrectHits()) / possible;
        }
    }

    public double precision() {
        int predicted = getPredictedHits();
        if (predicted == 0) {
            return 1.0;
        } else {
            return ((double) getCorrectHits()) / predicted;
        }
    }

    public static double harmonicMean(double p, double r) {
        double denom = p + r;
        if (denom == 0.0) {
            return 0.0;
        } else {
            return 2 * p * r / denom;
        }
    }

    public double f1() {
        return harmonicMean(precision(), recall());
    }

    public double accuracy() {
        int total = getTotal();
        if (total == 0) {
            return 1.0;
        } else {
            return ((double) getCorrect()) / getTotal();
        }
    }

    /**
     * rows correspond to the desired label; columns correspond to the predicted
     * label
     * 
     * & pred lab 1 & pred lab 2 & goldlab 1 & goldlab 2 &
     * 
     * @return
     */
    public String formatMatrix(Collection<L> keys) {
        return formatMatrix(keys, "  &  ", "  \\\\\n");
    }

    public static int[] getColWidths(String[][] table, int pad, int minWidth) {
        int nrows = table.length;
        int ncols = table[0].length;
        int colWidth[] = new int[ncols];
        // initialize the column widths
        for (int j = 0; j < ncols; j++) {
            colWidth[j] = Math.max(table[0][j].length() + pad, minWidth);
            for (int i = 1; i < nrows; i++) {
                colWidth[j] = Math.max(table[i][j].length() + pad, colWidth[j]);
            }
        }
        return colWidth;
    }

    public int getCount(L gold, L pred) {
        return goldPredPairCounts.get(new Pair<>(gold, pred));
    }

    public int getGoldCount(L gold) {
        return goldCounts.get(gold);
    }

    public int getPredCount(L pred) {
        return predCounts.get(pred);
    }

    public int getTotal() {
        return goldPredPairCounts.getTotal();
    }

    public String format(String name, Collection<L> labelOrder) {
        StringWriter sw = new StringWriter();
        sw.write("\n");
        // making it easier to sort by f1
        sw.write(String.format("   \t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
                               "name", "f1(prec)",
                               "majf1(majprec)",
                               "corhits", "majcorhits",
                               "psbhits", "total"));
        sw.write(String.format("~~~\t%s\t%.1f (%.1f)\t%.1f (%.1f)\t%s\t%s\t%s\t%s\n",
                               name, f1() * 100, precision() * 100,
                               majorityNonNilF1() * 100, majorityNonNilPrecision() * 100,
                               getCorrectHits(), majorityNonNilCorrectHits(),
                               getPossibleHits(), getTotal()));
        sw.write(String.format("==%s Precision: %s\n", name, precision()));
        sw.write(String.format("==%s Recall: %s\n", name, recall()));
        sw.write(String.format("==%s F1: %s\n", name, f1()));
        sw.write(String.format("==%s Accuracy: %s\n", name, accuracy()));
        sw.write(String.format("==%s MarjoirtyNonNilBaseline: %s\n", name, majorityNonNilF1()));
        sw.write(formatMatrix(labelOrder));
        return sw.toString();
    }

    public L majorityNonNilLabel() {
        L maxLabel = null;
        int maxCount = 0;
        for (L k : keys) {
            if (nilLabels.contains(k)) {
                continue;
            }
            int goldCount = goldCounts.get(k);
            if (goldCount >= maxCount) {
                maxCount = goldCount;
                maxLabel = k;
            }
        }
        return maxLabel;
    }

    public double majorityNonNilPrecision() {
        int possibleHits = getPossibleHits();
        // if possible hits is 0 then can get perfect precision and recall with nil labels
        if (possibleHits > 0) {
            // otherwise, all predictions will have to be non-nil and the best shot is the majority
            return ((double)majorityNonNilCorrectHits()) / getTotal();
        } else {
            return 1.0;
        }
    }

    public double majorityNonNilRecall() {
        // if possible hits is 0 then can get perfect precision and recall with nil labels
        int possibleHits = getPossibleHits();
        if (possibleHits > 0) {
            // otherwise, all predictions will have to be non-nil and the best shot is the majority
            return ((double)majorityNonNilCorrectHits()) / possibleHits;
        } else {
            return 1.0;
        }
    }

    public int majorityNonNilCorrectHits() {
        int possibleHits = getPossibleHits();
        if (possibleHits > 0) {
            return goldCounts.get(majorityNonNilLabel());
        } else {
            return 0;
        }
    }

    public double majorityNonNilF1() {
        return harmonicMean(majorityNonNilPrecision(), majorityNonNilRecall());
    }
    
    public List<String> getExamples(L gold, L pred) {
        return examples.get(new Pair<>(gold, pred));
    }

    public int numExamples(L gold, L pred) {
        List<String> examples = getExamples(gold, pred);
        if (examples == null) {
            return 0;
        } else {
            return examples.size();
        }
    }

    // TODO: add a precision row and a recall column to get label specific
    // precision and recall
    public String formatMatrix(Collection<L> keys, String cellSep, String lineSep) {

        // get the number of rows and columns
        List<L> rows = new ArrayList<L>(keys);
        List<L> cols = new ArrayList<L>(keys);
        DecimalFormat formatter = new DecimalFormat("#,###");
        int nrows = 2 + rows.size();
        int ncols = 2 + cols.size();

        // build an array of strings (including the headers and totals
        String formatted[][] = new String[nrows][ncols];

        formatted[0][0] = "gold \\ pred";
        formatted[0][cols.size() + 1] = "total";
        formatted[nrows - 1][ncols - 1] = formatter.format(getTotal());
        for (int j = 0; j < cols.size(); j++) {
            L predLabel = cols.get(j);
            // column heading
            formatted[0][j + 1] = predLabel.toString();
            // total predicted for jth label
            formatted[nrows - 1][j + 1] = formatter.format(getPredCount(predLabel));
        }

        formatted[rows.size() + 1][0] = "total";
        for (int i = 0; i < cols.size(); i++) {
            L goldLabel = rows.get(i);
            // row heading
            formatted[i + 1][0] = goldLabel.toString();
            // total for gold label
            formatted[i + 1][ncols - 1] = formatter.format(getGoldCount(goldLabel));

            // now fill in the columns for this gold row
            for (int j = 0; j < rows.size(); j++) {
                L predLabel = cols.get(j);
                int count = getCount(goldLabel, predLabel);
                formatted[i + 1][j + 1] = formatter.format(count);
            }
        }
        // compute the column widths
        // 0 pad, 0 min width
        int colWidth[] = getColWidths(formatted, 0, 0);

        // compute the total width (to help allocate space for the output string
        int totalWidth = 0;
        for (int j = 0; j < ncols; j++) {
            totalWidth += colWidth[j];
            if (j == ncols - 1) {
                totalWidth += lineSep.length();
            } else {
                totalWidth += cellSep.length();
            }
        }

        StringWriter sw = new StringWriter(totalWidth * nrows);
        for (int i = 0; i < nrows; i++) {
            for (int j = 0; j < ncols; j++) {
                // first make a format string with the width that we want
                String colWidthFormatString = String.format("%%%ds", colWidth[j]);
                sw.write(String.format(colWidthFormatString, formatted[i][j]));
                if (j == ncols - 1) {
                    sw.write(lineSep);
                } else {
                    sw.write(cellSep);
                }
            }
        }

        return sw.toString();

    }

    public Map<Pair<L, L>, List<String>> getExamples() {
        return examples;
    }

    public Set<L> keySet() {
        return keys;
    }

    public void print(String category, Collection<L> labelOrder, Writer out) throws IOException {
        out.write(format(category, labelOrder));
        // write out an example for each non-zero cell in the matrix
        for (Map.Entry<Pair<L, L>, List<String>> exList : getExamples().entrySet()) {
            for (String ex : exList.getValue()) {
                out.write(String.format("%s %s %s:\n%s\n", category, exList.getKey().get1().toString(),
                        exList.getKey().get2().toString(), ex));
            }
        }
    }
}
