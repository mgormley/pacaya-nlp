package edu.jhu.nlp.depparse;

import static edu.jhu.pacaya.gm.model.globalfac.LinkVar.FALSE;
import static edu.jhu.pacaya.gm.model.globalfac.LinkVar.TRUE;
import static edu.jhu.pacaya.gm.model.globalfac.LinkVar.TRUE_TRUE;

import java.util.Arrays;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.depparse.DepParseFactorGraphBuilder.DepParseFactorTemplate;
import edu.jhu.nlp.depparse.DepParseFactorGraphBuilder.GraFeTypedFactor;
import edu.jhu.pacaya.autodiff.Module;
import edu.jhu.pacaya.autodiff.Tensor;
import edu.jhu.pacaya.gm.inf.AbstractFgInferencer;
import edu.jhu.pacaya.gm.inf.Beliefs;
import edu.jhu.pacaya.gm.inf.FgInferencer;
import edu.jhu.pacaya.gm.inf.FgInferencerFactory;
import edu.jhu.pacaya.gm.model.Factor;
import edu.jhu.pacaya.gm.model.FactorGraph;
import edu.jhu.pacaya.gm.model.Factors;
import edu.jhu.pacaya.gm.model.ForwardOnlyFactorsModule;
import edu.jhu.pacaya.gm.model.Var;
import edu.jhu.pacaya.gm.model.VarSet;
import edu.jhu.pacaya.gm.model.VarTensor;
import edu.jhu.pacaya.gm.model.globalfac.GlobalFactor;
import edu.jhu.pacaya.gm.model.globalfac.LinkVar;
import edu.jhu.pacaya.gm.model.globalfac.ProjDepTreeFactor;
import edu.jhu.pacaya.hypergraph.Hyperalgo;
import edu.jhu.pacaya.hypergraph.Hyperalgo.HyperedgeDoubleFn;
import edu.jhu.pacaya.hypergraph.Hyperalgo.Scores;
import edu.jhu.pacaya.hypergraph.Hyperedge;
import edu.jhu.pacaya.hypergraph.Hypernode;
import edu.jhu.pacaya.hypergraph.depparse.DependencyScorer;
import edu.jhu.pacaya.hypergraph.depparse.ExplicitDependencyScorer;
import edu.jhu.pacaya.hypergraph.depparse.InsideOutsideDepParse;
import edu.jhu.pacaya.hypergraph.depparse.O2AllGraDpHypergraph;
import edu.jhu.pacaya.hypergraph.depparse.PCGBasicHypernode;
import edu.jhu.pacaya.util.collections.QLists;
import edu.jhu.pacaya.util.semiring.Algebra;
import edu.jhu.prim.arrays.DoubleArrays;
import edu.jhu.prim.arrays.IntArrays;

public class O2AllGraFgInferencer extends AbstractFgInferencer implements Module<Beliefs>, FgInferencer {

    public static class O2AllGraFgInferencerFactory implements FgInferencerFactory {

        private Algebra s;
        public O2AllGraFgInferencerFactory(Algebra s) {
            this.s = s;
        }
        
        @Override
        public FgInferencer getInferencer(FactorGraph fg) { 
            return new O2AllGraFgInferencer(fg, s);
        }

        @Override
        public Algebra getAlgebra() {
            return s;
        }
        
    }
    
    private static final Logger log = LoggerFactory.getLogger(O2AllGraFgInferencer.class);
    
    // Input:
    private Module<Factors> fm;
    private final Algebra s;
    private FactorGraph fg;
    private int n;
    // Output:
    private Beliefs b;
    private Beliefs bAdj;
    // Cached: 
    private O2AllGraDpHypergraph graph;
    private Scores sc;
    private int[][] ijFacs;
    private int[][][] ijkFacs;
    
    public O2AllGraFgInferencer(FactorGraph fg, Algebra s) {
        this(fg, ForwardOnlyFactorsModule.getFactorsModule(fg, s));
    }

    public O2AllGraFgInferencer(final FactorGraph fg, Module<Factors> fm) {
        this.s = fm.getAlgebra();
        this.fg = fg;
        this.fm = fm;
        // Guess the length of the sentence.
        // TODO: Pass this in to the constructor.
        n = -1;
        for (Var v : fg.getVars()) {
            LinkVar lv = (LinkVar) v;
            n = Math.max(n, lv.getChild()+1);
            n = Math.max(n, lv.getParent()+1);
        }
    }

    @Override
    public void run() {
        forward();
    }
    
    @Override
    public Beliefs forward() {
        // Create indices for looking up factors.
        gatherFactors();
        
        // Build the hypergraph.
        // - extract the edge and edge-pair scores from the factor graph.
        DependencyScorer scorer = forwardFgToScores();
        graph = new O2AllGraDpHypergraph(scorer, s, InsideOutsideDepParse.singleRoot);
        
        // Run inside-outside.
        sc = new Scores();
        Hyperalgo.forward(graph, graph.getPotentials(), s, sc);
        if (sc.beta[graph.getRoot().getId()] == s.zero()) {
            if (log.isTraceEnabled()) { printScoreDetails(); }
            throw new IllegalStateException("Scores disallowed all possible parses.");
        }
        if (log.isTraceEnabled()) { sc.prettyPrint(graph); }
        
        // Cache the beliefs.
        forwardBeliefs();
        return b;
    }

    @Override
    public void backward() {
        // Backprop from the beliefs to the hypernode marginals.
        backwardBeliefs();

        // Backprop from the hypernode marginals through the hyperedge weights to the edge-pair
        // scores.
        final Tensor scoresAdj = new Tensor(s, n+1, n+1, n+1);
        HyperedgeDoubleFn lambda = new HyperedgeDoubleFn() {            
            @Override
            public void apply(Hyperedge e, double adj_w_e) {
                if (e.getHeadNode() instanceof PCGBasicHypernode) {
                    PCGBasicHypernode h = (PCGBasicHypernode) e.getHeadNode();
                    scoresAdj.add(adj_w_e, h.getP()+1, h.getC()+1, h.getG()+1);
                }
            }
        };
        Hyperalgo.backward(graph, graph.getPotentials(), s, sc, lambda);
        
        // Backprop from the edge-pair scores to the factor modules. 
        backwardFgToScores(scoresAdj);
    }

    private DependencyScorer forwardFgToScores() {
        Factors factors = fm.getOutput();

        // Create the scores in the log semiring.
        double[][][] scores = new double[n+1][n+1][n+1];
        DoubleArrays.fill(scores, s.one());
        for (int i=0; i<n+1; i++) {
            for (int j=0; j<n+1; j++) {
                for (int g=0; g<n+1; g++) {
                    if (i <= g && g <= j && !(i==0 && g == O2AllGraDpHypergraph.NIL)) { continue; }
                    if (ijkFacs[i][j][g] != -1) {
                        VarTensor psi_ijk = factors.get(ijkFacs[i][j][g]);
                        scores[i][j][g] = s.times(scores[i][j][g], psi_ijk.getValue(TRUE_TRUE));
                    }
                    if (ijFacs[i][j] != -1) {
                        VarTensor psi_ij = factors.get(ijFacs[i][j]);
                        scores[i][j][g] = s.times(scores[i][j][g], psi_ij.getValue(TRUE));
                    }
                }
            }
        }
        
        if (log.isTraceEnabled()) { log.trace("scores: " + Arrays.deepToString(scores)); }
        return new ExplicitDependencyScorer(scores, n);
    }

    private void backwardFgToScores(Tensor scoresAdj) {
        Factors factors = fm.getOutput();
        Factors factorsAdj = fm.getOutputAdj();

        // Create the scores in the log semiring.
        for (int i=0; i<n+1; i++) {
            for (int j=0; j<n+1; j++) {
                for (int g=0; g<n+1; g++) {
                    if (i <= g && g <= j && !(i==0 && g == O2AllGraDpHypergraph.NIL)) { continue; }
                    VarTensor psi_ijk = null;
                    VarTensor psi_ij = null;
                    VarTensor adj_ijk = null;
                    VarTensor adj_ij = null;
                    double psi_ijk_tt = s.one();
                    double psi_ij_t = s.one();
                    if (ijkFacs[i][j][g] != -1) {
                        psi_ijk = factors.get(ijkFacs[i][j][g]);
                        adj_ijk = factorsAdj.get(ijkFacs[i][j][g]);
                        psi_ijk_tt = psi_ijk.getValue(TRUE_TRUE);
                    }
                    if (ijFacs[i][j] != -1) {
                        psi_ij = factors.get(ijFacs[i][j]);
                        adj_ij = factorsAdj.get(ijFacs[i][j]);
                        psi_ij_t = psi_ij.getValue(TRUE);
                    }
                    if (adj_ijk != null) {
                        adj_ijk.addValue(TRUE_TRUE, s.times(scoresAdj.get(i,j,g), psi_ij_t));
                    }
                    if (adj_ij != null) {
                        adj_ij.addValue(TRUE, s.times(scoresAdj.get(i,j,g), psi_ijk_tt));
                    }
                }
            }
        }
    }

    /** Computes the beliefs from the hypernode marginals. */
    protected void forwardBeliefs() {        
        b = new Beliefs(new VarTensor[fg.getNumVars()], new VarTensor[fg.getNumFactors()]);
        for (int v=0; v<b.varBeliefs.length; v++) {
            b.varBeliefs[v] = calcVarBeliefs(fg.getVar(v));
        }
        for (int a=0; a<b.facBeliefs.length; a++) {
            Factor f = fg.getFactor(a);
            if (f instanceof GlobalFactor) { continue; }
            b.facBeliefs[a] = calcFactorBeliefs(f);
        }
    }

    /** Backprops the computation of beliefs from hypernode marginals. */
    private void backwardBeliefs() {
        sc.marginalAdj = new double[sc.marginal.length];
        for (int a=0; a<b.facBeliefs.length; a++) {
            Factor f = fg.getFactor(a);
            if (f instanceof GlobalFactor) { continue; }
            backwardFactorBeliefs(f);
        }
        for (int v=0; v<b.varBeliefs.length; v++) {
            backwardVarBeliefs(fg.getVar(v));
        }
    }

    // Variable Marginals:
    // p(x_{ij} = T) = \sum_{k=1}^K p(e_{ijk})
    // p(x_{ij} = F) = 1.0 - p(x_{ij} = T)
    private VarTensor calcVarBeliefs(Var var) {
        LinkVar lv = (LinkVar)var;
        int i = lv.getParent()+1;
        int j = lv.getChild()+1;
        Hypernode[][][][] c = graph.getChart();
        double lv1 = s.zero();
        for (int g=0; g<n+1; g++) {
            if (c[i][j][g][O2AllGraDpHypergraph.INCOMPLETE] != null) {
                int id = c[i][j][g][O2AllGraDpHypergraph.INCOMPLETE].getId();                
                lv1 = s.plus(lv1, sc.marginal[id]);
            }
        }
        double lv0 = s.minus(s.one(), lv1);
        VarTensor b = new VarTensor(s, new VarSet(var));
        b.setValue(TRUE, lv1);
        b.setValue(FALSE, lv0);
        return b;
    }
    
    // Variable Marginal Adjoints:
    // <== p(x_{ij} = F) = 1.0 - p(x_{ij} = T)        
    // \adj{p(x_{ij} = T) += \adj{p(x_{ij} = F)} * -1.0
    //
    // <== p(x_{ij} = T) = \sum_{k=1}^K p(e_{ijk})
    // \adj{p(e_{ijk})} += \adj{p(x_{ij} = T)} \forall k
    private void backwardVarBeliefs(Var var) {
        LinkVar lv = (LinkVar)var;
        int i = lv.getParent()+1;
        int j = lv.getChild()+1;
        int v = var.getId();
        Hypernode[][][][] c = graph.getChart();
        VarTensor pxAdj = bAdj.varBeliefs[v];
        pxAdj.add(s.negate(pxAdj.get(FALSE)), TRUE);
        for (int g=0; g<n+1; g++) {
            if (c[i][j][g][O2AllGraDpHypergraph.INCOMPLETE] != null) {
                int id = c[i][j][g][O2AllGraDpHypergraph.INCOMPLETE].getId();
                sc.marginalAdj[id] = s.plus(sc.marginalAdj[id], pxAdj.get(TRUE));
            }
        }
    }

    // Factor Marginals:
    // p(x_{ijk} = TT) = p(e_{ijk})
    // p(x_{ijk} = FT) = p(x_{ij} = T) - p(e_{ijk})
    // p(x_{ijk} = TF) = p(x_{ki} = T) - p(e_{ijk})
    // p(x_{ijk} = FF) = p(x_{ki} = T) - p(x_{ijk} = FT)
    private VarTensor calcFactorBeliefs(Factor f) {
        if (f instanceof GraFeTypedFactor && ((GraFeTypedFactor) f).getFactorType() == DepParseFactorTemplate.GRANDPARENT) {
            GraFeTypedFactor ff = (GraFeTypedFactor) f;
            int k = ff.g+1;
            int i = ff.p+1;
            int j = ff.c+1;                        
            VarTensor b = new VarTensor(s, f.getVars());
            int id = graph.getChart()[i][j][k][O2AllGraDpHypergraph.INCOMPLETE].getId();            
            b.set(sc.marginal[id], TRUE, TRUE);
            // Compute the other marginals using the variable marginals. Consider the 2x2 table of 
            // probabilities. We have the marginals for all rows and columns, plus one entry.
            VarTensor be0 = getVarBeliefs(f.getVars().get(0));
            VarTensor be1 = getVarBeliefs(f.getVars().get(1));
            b.set(carefulMinus(s, be1.get(TRUE), b.get(TRUE, TRUE)), FALSE, TRUE);
            b.set(carefulMinus(s, be0.get(TRUE), b.get(TRUE, TRUE)), TRUE, FALSE);
            b.set(carefulMinus(s, be0.get(FALSE), b.get(FALSE, TRUE)), FALSE, FALSE);
            log.trace(String.format("p=%d c=%d g=%d b=%s", i, j, k, b.toString()));
            return b;
        } else if (f.getVars().size() == 1 && f.getVars().get(0) instanceof LinkVar) {
            return getVarBeliefs(f.getVars().get(0));
        } else if (f.getVars().size() == 0) {
            return new VarTensor(s, new VarSet());
        } else {
            throw new RuntimeException("Unsupported factor type: " + f.getClass());
        }
    }

    // Factor Marginal Adjoints:
    // 1. p(x_{ijk} = FF) = p(x_{ki} = T) - p(x_{ijk} = FT)        
    // \adj{p(x_{ijk} = FT)} += \adj{p(x_{ijk} = FF)} * -1.0
    // \adj{p(x_{ki} = T)} += \adj{p(x_{ijk} = FF)}
    //
    // 2. p(x_{ijk} = TF) = p(x_{ki} = T) - p(e_{ijk})
    // \adj{p(e_{ijk})} += \adj{p(x_{ijk} = TF)} * -1.0
    // \adj{p(x_{ki} = T)} += \adj{p(x_{ijk} = TF)}
    //
    // 3. p(x_{ijk} = FT) = p(x_{ij} = T) - p(e_{ijk})
    // \adj{p(e_{ijk})} += \adj{p(x_{ijk} = FT)} * -1.0
    // \adj{p(x_{ij} = T)} += \adj{p(x_{ijk} = FT)}
    //
    // 4. p(x_{ijk} = TT) = p(e_{ijk})
    // \adj{p(e_{ijk})} += \adj{p(x_{ijk} = TT)}
    private void backwardFactorBeliefs(Factor f) {
        if (f instanceof GraFeTypedFactor && ((GraFeTypedFactor) f).getFactorType() == DepParseFactorTemplate.GRANDPARENT) {
            GraFeTypedFactor ff = (GraFeTypedFactor) f;
            int k = ff.g+1;
            int i = ff.p+1;
            int j = ff.c+1;                        
            LinkVar v0 = (LinkVar) ff.getVars().get(0);
            LinkVar v1 = (LinkVar) ff.getVars().get(1);
            int v_ij = (v0.getParent() == ff.p && v0.getChild() == ff.c) ? v0.getId() : v1.getId();
            int v_ki = (v0.getParent() == ff.p && v0.getChild() == ff.c) ? v1.getId() : v0.getId();
            VarTensor adj_px_ij = bAdj.varBeliefs[v_ij];
            VarTensor adj_px_ki = bAdj.varBeliefs[v_ki];
            VarTensor adj_px_ijk = bAdj.facBeliefs[ff.getId()];
            int id = graph.getChart()[i][j][k][O2AllGraDpHypergraph.INCOMPLETE].getId();

            // 1. 
            adj_px_ijk.add(s.negate(adj_px_ijk.get(FALSE, TRUE)), FALSE, TRUE);
            adj_px_ki.add(adj_px_ijk.get(FALSE, FALSE));            
            // 2.
            sc.marginalAdj[id] = s.plus(sc.marginalAdj[id], s.negate(adj_px_ijk.get(TRUE, FALSE)));
            adj_px_ki.add(adj_px_ijk.get(TRUE, FALSE), TRUE);
            // 3.
            sc.marginalAdj[id] = s.plus(sc.marginalAdj[id], s.negate(adj_px_ijk.get(FALSE, TRUE)));
            adj_px_ij.add(adj_px_ijk.get(FALSE, TRUE), TRUE);
            // 4.
            sc.marginalAdj[id] = s.plus(sc.marginalAdj[id], adj_px_ijk.get(TRUE, TRUE));

        } else if (f.getVars().size() == 1 && f.getVars().get(0) instanceof LinkVar) {
            // Just add this adjoint to the corresponding variable belief adjoint.
            int v = f.getVars().get(0).getId();
            bAdj.varBeliefs[v].add(bAdj.facBeliefs[f.getId()]);
        } else if (f.getVars().size() == 0) {
            // Nothing to do.
        } else {
            throw new RuntimeException("Unsupported factor type: " + f.getClass());
        }
    }

    /** Creates indices for looking up factors: 
     * 1. graFacs maps i,j,k indices to a grandparent factor ID
     * 2. edgeFacs maps i,j indices to an edge factor ID.
     */
    private void gatherFactors() {
        ijFacs = new int[n+1][n+1];
        ijkFacs = new int[n+1][n+1][n+1];
        IntArrays.fill(ijFacs, -1);
        IntArrays.fill(ijkFacs, -1);
        
        boolean containsProjDepTreeConstraint = false;
        for (int a=0; a<fg.getNumFactors(); a++) {
            Factor f = fg.getFactor(a);
            if (f instanceof ProjDepTreeFactor) {
                containsProjDepTreeConstraint = true;
            } else if (f instanceof GraFeTypedFactor && ((GraFeTypedFactor) f).getFactorType() == DepParseFactorTemplate.GRANDPARENT) {
                GraFeTypedFactor ff = (GraFeTypedFactor) f;
                ijkFacs[ff.p+1][ff.c+1][ff.g+1] = ff.getId();
            } else if (f.getVars().size() == 1 && f.getVars().get(0) instanceof LinkVar) {
                LinkVar lv = (LinkVar) f.getVars().get(0);
                int i = lv.getParent() + 1;
                int j = lv.getChild() + 1;
                ijFacs[i][j] = f.getId();
            } else if (f.getVars().size() == 0) {
                // Ignore clamped factor.
            } else {
                throw new RuntimeException("Unsupported factor type: " + f.getClass());
            }
        }
        if (!containsProjDepTreeConstraint) {
            throw new IllegalStateException("This inference method is only applicable to factor graphs containing "
                        + " a factor constraining to a projective dependency tree.");
        }
    }

    /** Returns a-b or 0 if b > a. This is to address floating point issues. */
    private double carefulMinus(Algebra s, double a, double b) {
        if (s.gt(b, a)) {
            return s.zero();
        } else {
            return s.minus(a, b);
        }
    }

    protected void printScoreDetails() {
        int i;
        i=0;
        for (Factor f : fg.getFactors()) {
            if (f.getVars().size() == 1) {
                LinkVar lv = (LinkVar) f.getVars().get(0);
                int p = lv.getParent();
                int c = lv.getChild();
                if (f.getLogUnormalizedScore(TRUE) == 0.0) {
                    System.out.printf("p=%d c=%d lv=TRUE sc=%f f=%d\n", p, c, f.getLogUnormalizedScore(TRUE), i);
                }
            }
            i++;
        }
        i=0;
        for (Factor f : fg.getFactors()) {
            if (f.getVars().size() == 1) {
                LinkVar lv = (LinkVar) f.getVars().get(0);
                int p = lv.getParent();
                int c = lv.getChild();
                if (f.getLogUnormalizedScore(TRUE) != 0.0) {
                    System.out.printf("p=%d c=%d lv=FALSE sc=%f f=%d \n", p, c, f.getLogUnormalizedScore(TRUE), i);
                }
            }
            i++;
        }
    }
    
    @Override
    protected VarTensor getVarBeliefs(Var var) {
        return getVarBeliefs(var.getId());
    }
    
    @Override
    protected VarTensor getFactorBeliefs(Factor factor) {
        return getFactorBeliefs(factor.getId());
    }
    
    private VarTensor getVarBeliefs(int varId) {
        return b.varBeliefs[varId];
    }
    
    private VarTensor getFactorBeliefs(int facId) {
        if (b.facBeliefs[facId] == null) {
            // Beliefs for global factors are not cached.
            Factor factor = fg.getFactor(facId);
            assert factor instanceof GlobalFactor;
            VarTensor b = calcFactorBeliefs(factor);
            b.normalize();
            return b;
        }
        return b.facBeliefs[facId];
    }

    @Override
    public double getPartitionBelief() {
        return sc.beta[graph.getRoot().getId()];
    }

    @Override
    public FactorGraph getFactorGraph() {
        return fg;
    }

    @Override
    public Algebra getAlgebra() {
        return s;
    }

    @Override
    public Beliefs getOutput() {
        return b;
    }

    @Override
    public Beliefs getOutputAdj() {
        if (bAdj == null) {            
            bAdj = b.copyAndFill(s.zero());
        }
        return bAdj;
    }

    @Override
    public void zeroOutputAdj() {
        if (bAdj != null) { bAdj.fill(s.zero()); }
    }

    @Override
    public List<Module<Factors>> getInputs() {
        return QLists.getList(fm);
    }

}
