package edu.jhu.nlp.sprl;

import java.io.StringWriter;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import edu.jhu.prim.tuple.Pair;

public class ConfusionMatrix<L> {

    /**
     * labeled confusion matrix
     */
    Counter<Pair<L, L>> goldPredPairCounts;
    Counter<L> goldCounts;
    Counter<L> predCounts;
    Set<L> keys;
    L nilLabel;

    /**
     * unlabeled confusion matrix (coarsen labels to binary based on whether or
     * not they match the nilLabel)
     */
    // Counter<Pair<Boolean, Boolean>> predGoldHitPaircounts;

    public ConfusionMatrix() {
        this(null);
    }

    public ConfusionMatrix(L nilLabel) {
        this.nilLabel = nilLabel;
        goldPredPairCounts = new Counter<>();
        goldCounts = new Counter<>();
        predCounts = new Counter<>();
        keys = new HashSet<>();
    }

    public void recordPrediction(L gold, L pred) {
        // count the pair
        goldPredPairCounts.add(new Pair<>(gold, pred));
        goldCounts.add(gold);
        predCounts.add(pred);
        // keep track of keys
        keys.add(gold);
        keys.add(pred);
    }

    public int getCorrect() {
        int total = 0;
        for (L k : goldCounts.keySet()) {
            total += getCount(k, k);
        }
        return total;
    }

    public int getCorrectHits() {
        return getCorrect() - getCount(nilLabel, nilLabel);
    }
    
    public double recall() {
        int possible = getTotal() - getGoldCount(nilLabel);
        return ((double) getCorrectHits()) / possible; 
    }

    public double precision() {
        int predicted = getTotal() - getPredCount(nilLabel);
        return ((double) getCorrectHits()) / predicted; 
    }

    public double f1() {
        double p = precision();
        double r = recall();
        if (p == 0.0 && r == 0.0) {
            return 0.0;
        } else {
            return 2 * p * r / (p + r);
        }
    }
    
    public double accuracy() {
        return ((double) getCorrect()) / getTotal(); 
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
        sw.write(String.format("==%s==\n", name));
        sw.write(String.format("Accuracy: %s\n", accuracy()));
        sw.write(String.format("Precision: %s\n", precision()));
        sw.write(String.format("Recall: %s\n", recall()));
        sw.write(String.format("F1: %s\n", f1()));
        sw.write(formatMatrix(labelOrder));
        return sw.toString();
    }

    public String formatMatrix(Collection<L> keys, String cellSep, String lineSep) {

        // get the number of rows and columns
        List<L> rows = new ArrayList<L>(keys == null ? goldCounts.keySet() : keys);
        List<L> cols = new ArrayList<L>(keys == null ? predCounts.keySet() : keys);
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

    public Set<L> keySet() {
        return keys;
    }
}
