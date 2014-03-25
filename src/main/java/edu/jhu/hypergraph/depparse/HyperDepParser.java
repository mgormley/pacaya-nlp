package edu.jhu.hypergraph.depparse;

import edu.jhu.hypergraph.Hyperalgo;
import edu.jhu.hypergraph.Hyperalgo.Scores;
import edu.jhu.hypergraph.Hyperedge;
import edu.jhu.hypergraph.Hypernode;
import edu.jhu.hypergraph.Hyperpotential;
import edu.jhu.hypergraph.HyperpotentialFoe;
import edu.jhu.parse.dep.ProjectiveDependencyParser.DepIoChart;
import edu.jhu.parse.dep.ProjectiveDependencyParser.DepParseChart;
import edu.jhu.parse.dep.ProjectiveDependencyParser.DepParseType;
import edu.jhu.prim.tuple.Pair;
import edu.jhu.prim.util.math.FastMath;
import edu.jhu.util.semiring.LogPosNegSemiring;
import edu.jhu.util.semiring.LogSemiring;
import edu.jhu.util.semiring.Semiring;
import edu.jhu.util.semiring.SemiringExt;
import edu.jhu.util.semiring.Semirings;

public class HyperDepParser {

    /**
     * First-order expectation semiring potential function used to compute
     * entropy. See section 6.1 of Li & Eisner (2009).
     * 
     * @author mgormley
     */
    public static final class EntropyHyperpotentialFoe implements HyperpotentialFoe {
        private final Hyperpotential w;

        private EntropyHyperpotentialFoe(Hyperpotential w) {
            this.w = w;
        }

        @Override
        public double getScore(Hyperedge e, Semiring s) {
            return w.getScore(e, s);
        }

        @Override
        public double getScoreFoe(Hyperedge e, SemiringExt s) {
            double p_e = w.getScore(e, s);
            if (p_e == s.zero()) {
                return p_e;
            }
            return s.times(p_e, s.fromReal(FastMath.log(s.toReal(p_e))));
        }
    }

    /**
     * Runs the inside-outside algorithm for dependency parsing.
     * 
     * @param fracRoot Input: The edge weights from the wall to each child.
     * @param fracChild Input: The edge weights from parent to child.
     * @return The parse chart.
     */
    public static DepIoChart insideOutsideAlgorithm(double[] fracRoot, double[][] fracChild) {
        // Currently we only support this semiring since DepParseChart assumes log probs.
        LogSemiring semiring = new LogSemiring();
        //LogPosNegSemiring semiring = new LogPosNegSemiring();
        //Semirings.fromLogProb(fracRoot, semiring);
        //Semirings.fromLogProb(fracChild, semiring);
        
        FirstOrderDepParseHypergraph graph = new FirstOrderDepParseHypergraph(fracRoot, fracChild, semiring);
        Scores scores = new Scores();
        Hyperalgo.insideAlgorithm(graph, graph.getPotentials(), semiring, scores);
        Hyperalgo.outsideAlgorithm(graph, graph.getPotentials(), semiring, scores);

        return getDepIoChart(graph, scores);
    }

    public static DepIoChart getDepIoChart(FirstOrderDepParseHypergraph graph, Scores scores) {
        final int n = graph.getNumTokens();        
        final DepParseChart inChart = new DepParseChart(n, DepParseType.INSIDE);
        final DepParseChart outChart = new DepParseChart(n, DepParseType.INSIDE);
        Hypernode[] wallChart = graph.getWallChart();
        Hypernode[][][][] childChart = graph.getChildChart();
        for (int width = 1; width < n; width++) {
            for (int s = 0; s < n - width; s++) {
                int t = s + width;                
                for (int d=0; d<2; d++) {
                    for (int c=0; c<2; c++) {                        
                        int id = childChart[s][t][d][c].getId();
                        inChart.updateCell(s, t, d, c, scores.beta[id], -1);
                        outChart.updateCell(s, t, d, c, scores.alpha[id], -1);
                    }
                }
            }
        }
        for (int s=0; s<n; s++) {
            int id = wallChart[s].getId();
            inChart.updateGoalCell(s, scores.beta[id]);
            outChart.updateGoalCell(s, scores.alpha[id]);
        }
        // Root is automatically updated by updateGoalCell() calls above.        
        return new DepIoChart(inChart, outChart);
    }
    
    /**
     * Runs the inside-outside algorithm for dependency parsing.
     * 
     * @param fracRoot Input: The edge weights from the wall to each child.
     * @param fracChild Input: The edge weights from parent to child.
     * @return The parse chart.
     */
    public static Pair<FirstOrderDepParseHypergraph, Scores> insideAlgorithmEntropyFoe(double[] fracRoot, double[][] fracChild) {
        final SemiringExt semiring = new LogPosNegSemiring();         
        return insideAlgorithmEntropyFoe(fracRoot, fracChild, semiring);
    }

    public static Pair<FirstOrderDepParseHypergraph, Scores> insideAlgorithmEntropyFoe(double[] fracRoot,
            double[][] fracChild, final SemiringExt semiring) {
        Semirings.fromLogProb(fracRoot, semiring);
        Semirings.fromLogProb(fracChild, semiring);
        
        FirstOrderDepParseHypergraph graph = new FirstOrderDepParseHypergraph(fracRoot, fracChild, semiring);
        Scores scores = new Scores();
        final Hyperpotential w = graph.getPotentials();
        HyperpotentialFoe wFoe = new EntropyHyperpotentialFoe(w);
        Hyperalgo.insideAlgorithmFirstOrderExpect(graph, wFoe, semiring, scores);

        return new Pair<FirstOrderDepParseHypergraph, Scores>(graph, scores);
    }
    
}