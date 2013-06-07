package edu.jhu.hltcoe.parse.cky.chart;

import edu.jhu.hltcoe.parse.cky.BinaryTree;
import edu.jhu.hltcoe.parse.cky.CnfGrammar;
import edu.jhu.hltcoe.parse.cky.Rule;
import edu.jhu.hltcoe.util.Pair;


/**
 * Parsing chart that stores every cell explicitly.
 * 
 * @author mgormley
 * 
 */
public class Chart {
    
    public static class BackPointer {
        private Rule r;
        private int mid;

        public BackPointer(Rule r, int mid) {
            this.r = r;
            this.mid = mid;
        }

        /**
         * Get the midpoint of the rule application.
         */
        public int getMid() {
            return mid;
        }
        
        /**
         * Get the rule.
         */
        public Rule getRule() {
            return r;
        }
    }

    public enum ChartCellType { FULL, SINGLE_HASH, DOUBLE_HASH, FULL_BREAK_TIES };
    public enum ParseType { VITERBI, INSIDE };
    
    private final ChartCellType cellType;
    private final CnfGrammar grammar;
    
    private ChartCell[][] chart;
    private int[] sent;
    private ParseType parseType;

    public Chart(int[] sent, CnfGrammar grammar, ChartCellType cellType, ParseType parseType) {
        this.cellType = cellType;
        this.parseType = parseType;
        this.sent = sent;
        this.grammar = grammar;
        this.chart = getNewChart(sent, grammar, cellType, parseType);
    }
    
    /**
     * Resets the chart for the input sentence.
     */
    public void reset(int[] sent) {
        this.sent = sent;
        // Ensure that the chart is large enough.
        if (sent.length > chart.length){
            chart = getNewChart(sent, grammar, cellType, parseType);
        } else {
            // Clear the chart.
            //
            // Note that we only need to clear the portion that will be used while parsing this sentence.
            for (int i = 0; i < sent.length; i++) {
                for (int j = i+1; j < sent.length + 1; j++) {
                    chart[i][j].reset();
                }
            }
        }
    }
    
    /**
     * Gets a new chart of the appropriate size for the sentence, specific to
     * this grammar, and with cells of the specified type.
     */
    private static ChartCell[][] getNewChart(int[] sent, CnfGrammar grammar, ChartCellType cellType, ParseType parseType) {
        ChartCell[][] chart = new ChartCell[sent.length][sent.length+1];
        for (int i = 0; i < chart.length; i++) {
            for (int j = i+1; j < chart[i].length; j++) {
                if (parseType == ParseType.INSIDE && cellType != ChartCellType.FULL) {
                    throw new RuntimeException("Inside algorithm not implemented for cell type: " + cellType);
                }
                switch(cellType) {
                case SINGLE_HASH:
                    chart[i][j] = new SingleHashChartCell(grammar);
                    break;
                case DOUBLE_HASH:
                    chart[i][j] = new DoubleHashChartCell(grammar);
                    break;
                case FULL:
                    chart[i][j] = new FullChartCell(grammar, parseType);
                    break;
                case FULL_BREAK_TIES:
                    chart[i][j] = new FullTieBreakerChartCell(grammar, true);
                    break;
                default:
                    throw new RuntimeException("not implemented for " + cellType);
                }                
            }
        }
        return chart;
    }    

    public Pair<BinaryTree,Double> getViterbiParse() {
        BinaryTree root = getViterbiTree(0, sent.length, grammar.getRootSymbol());
        double rootScore = chart[0][sent.length].getScore(grammar.getRootSymbol());
        return new Pair<BinaryTree, Double>(root, rootScore);
    }
    
    /**
     * Gets the highest probability tree with the span (start, end) and the root symbol rootSymbol.
     * 
     * @param start The start of the span of the requested tree.
     * @param end The end of the span of the requested tree.
     * @param rootSymbol The symbol of the root of the requested tree.
     * @return 
     */
    private BinaryTree getViterbiTree(int start, int end, int rootSymbol) {
        ChartCell cell = chart[start][end];
        BackPointer bp = cell.getBp(rootSymbol);
        if (bp == null) {
            return null;
        }
        
        BinaryTree leftChild;
        BinaryTree rightChild;
        if (bp.r.isLexical()) {
            leftChild = new BinaryTree(bp.r.getLeftChild(), start, end, null, null, true, grammar.getLexAlphabet());
            rightChild = null;
        } else if (bp.r.isUnary()) {
            leftChild = getViterbiTree(start, bp.mid, bp.r.getLeftChild());
            rightChild = null;
        } else {
            leftChild = getViterbiTree(start, bp.mid, bp.r.getLeftChild());
            rightChild = getViterbiTree(bp.mid, end, bp.r.getRightChild());
        }
        
        return new BinaryTree(rootSymbol, start, end, leftChild, rightChild, false, grammar.getNtAlphabet());
    }

    public ChartCell getCell(int start, int end) {
        return chart[start][end];
    }
    
    public CnfGrammar getGrammar() {
        return grammar;
    }
}
