package edu.jhu.nlp.depparse;

import static org.junit.Assert.assertEquals;

import java.util.List;

import org.junit.After;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;

import edu.jhu.nlp.FeTypedFactor;
import edu.jhu.nlp.data.DepEdgeMask;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.depparse.DepParseFactorGraphBuilder.DepParseFactorGraphBuilderPrm;
import edu.jhu.nlp.depparse.DepParseFactorGraphBuilder.GraFeTypedFactor;
import edu.jhu.pacaya.autodiff.AbstractModuleTest;
import edu.jhu.pacaya.autodiff.AbstractModuleTest.OneToOneFactory;
import edu.jhu.pacaya.autodiff.Module;
import edu.jhu.pacaya.gm.feat.FeatureExtractor;
import edu.jhu.pacaya.gm.feat.FeatureVector;
import edu.jhu.pacaya.gm.inf.BeliefPropagationTest;
import edu.jhu.pacaya.gm.inf.Beliefs;
import edu.jhu.pacaya.gm.inf.BruteForceInferencer;
import edu.jhu.pacaya.gm.model.Factor;
import edu.jhu.pacaya.gm.model.FactorGraph;
import edu.jhu.pacaya.gm.model.Factors;
import edu.jhu.pacaya.gm.model.FactorsModule;
import edu.jhu.pacaya.gm.model.FeExpFamFactor;
import edu.jhu.pacaya.gm.model.FgModel;
import edu.jhu.pacaya.gm.model.FgModelIdentity;
import edu.jhu.pacaya.gm.model.Var;
import edu.jhu.pacaya.gm.model.Var.VarType;
import edu.jhu.pacaya.gm.model.VarConfig;
import edu.jhu.pacaya.gm.model.globalfac.LinkVar;
import edu.jhu.pacaya.gm.train.SimpleVCFeatureExtractor;
import edu.jhu.pacaya.hypergraph.depparse.InsideOutsideDepParse;
import edu.jhu.pacaya.util.FeatureNames;
import edu.jhu.pacaya.util.collections.QLists;
import edu.jhu.pacaya.util.semiring.Algebra;
import edu.jhu.pacaya.util.semiring.LogSemiring;
import edu.jhu.pacaya.util.semiring.RealAlgebra;
import edu.jhu.prim.tuple.Pair;

public class O2AllGraFgInferencerTest {
    
    private boolean oldSingleRoot;

    @Before
    public void setUp() {
        oldSingleRoot = InsideOutsideDepParse.singleRoot;
    }
    
    @After
    public void tearDown() {
        InsideOutsideDepParse.singleRoot = oldSingleRoot;
    }
    
    @Test
    public void testZeroModelSingleRoot() {
        InsideOutsideDepParse.singleRoot = true;
        checkBruteForceEqualsDynamicProgramming(true, QLists.getList("a"));
        checkBruteForceEqualsDynamicProgramming(true, QLists.getList("a", "b"));
        checkBruteForceEqualsDynamicProgramming(true, QLists.getList("a", "b", "c"));
        checkBruteForceEqualsDynamicProgramming(true, QLists.getList("a", "b", "c", "d"));
    }
    
    @Test
    public void testZeroModelMultiRoot() {
        InsideOutsideDepParse.singleRoot = false;
        checkBruteForceEqualsDynamicProgramming(true, QLists.getList("a"));
        checkBruteForceEqualsDynamicProgramming(true, QLists.getList("a", "b"));
        checkBruteForceEqualsDynamicProgramming(true, QLists.getList("a", "b", "c"));
        checkBruteForceEqualsDynamicProgramming(true, QLists.getList("a", "b", "c", "d"));
    }
    
    @Test
    public void testNonzeroModelSingleRoot() {
        InsideOutsideDepParse.singleRoot = true;
        checkBruteForceEqualsDynamicProgramming(false, QLists.getList("a"));
        checkBruteForceEqualsDynamicProgramming(false, QLists.getList("a", "b"));
        // In the tests below, O2AllGraFgInferencer thinks an non-projective configuration of the
        // variables has non-zero probability due to floating point precision issues.
        checkBruteForceEqualsDynamicProgramming(false, QLists.getList("a", "b", "c"), LogSemiring.getInstance());
        //checkBruteForceEqualsDynamicProgramming(false, QLists.getList("a", "b", "c", "d"), LogSemiring.getInstance());
    }
    
    @Test
    public void testNonzeroModelMultiRoot() {
        InsideOutsideDepParse.singleRoot = false;
        checkBruteForceEqualsDynamicProgramming(false, QLists.getList("a"));
        checkBruteForceEqualsDynamicProgramming(false, QLists.getList("a", "b"));
        checkBruteForceEqualsDynamicProgramming(false, QLists.getList("a", "b", "c"));
        checkBruteForceEqualsDynamicProgramming(false, QLists.getList("a", "b", "c", "d"));
    }
    
    private static void checkBruteForceEqualsDynamicProgramming(boolean zeroModel, List<String> words) {
        checkBruteForceEqualsDynamicProgramming(zeroModel, words, RealAlgebra.getInstance());
        checkBruteForceEqualsDynamicProgramming(zeroModel, words, LogSemiring.getInstance());
    }
    
    private static void checkBruteForceEqualsDynamicProgramming(boolean zeroModel, List<String> words, Algebra s) {
        FactorGraph fg = getO2AllGraFg(zeroModel, words);
        
        BruteForceInferencer bf = new BruteForceInferencer(fg, s);
        bf.run();
        O2AllGraFgInferencer dp = new O2AllGraFgInferencer(fg, s);
        dp.run();
        
        if (words.size() <= 3) {
            System.out.println("joint: "+bf.getJointFactor().toString(true));
        }

        double tolerance = 1e-5;
        // Scale is too large: assertEquals(bf.getPartition(), dp.getPartition(), tolerance);
        assertEquals(bf.getLogPartition(), dp.getLogPartition(), tolerance);
        //assertEqualMarginals(fg, bf, dp, tolerance, false);
        BeliefPropagationTest.assertEqualMarginals(fg, bf, dp, tolerance, false);
    }

    private static FactorGraph getO2AllGraFg(boolean zeroModel, List<String> words) {
        return getO2AllGraFgAndModel(zeroModel, words).get1();
    }
    
    private static Pair<FactorGraph,FgModel> getO2AllGraFgAndModel(boolean zeroModel, List<String> words) {
        DepParseFactorGraphBuilderPrm prm = new DepParseFactorGraphBuilderPrm();
        prm.useProjDepTreeFactor = true;
        prm.grandparentFactors = true;
        prm.arbitrarySiblingFactors = false;
        prm.linkVarType = VarType.PREDICTED;
        DepParseFactorGraphBuilder builder = new DepParseFactorGraphBuilder(prm);
        FactorGraph fg = new FactorGraph();
        
        DepEdgeMask depEdgeMask = new DepEdgeMask(words.size(), true);
        AnnoSentence sent = new AnnoSentence();
        sent.setWords(words);
        sent.setDepEdgeMask(depEdgeMask);

        FeatureNames alphabet = new FeatureNames();
        LinkVarFe fe = new LinkVarFe(alphabet);
        builder.build(sent, fe, fg);
        
        FgModel model = new FgModel(1000);
        if (!zeroModel) {
            for (int i=0; i<model.getNumParams(); i++) {
                model.getParams().set(i, Math.log(i+2));// (i*31 % 1009)/100.);
            }
            //model.setRandomStandardNormal();
        }
        fg.updateFromModel(model);
        System.out.println("Factors:");
        for (Factor f : fg.getFactors()) {
            System.out.println(f);
        }
        return new Pair<>(fg, model);
    }
    
    private static class LinkVarFe extends SimpleVCFeatureExtractor implements FeatureExtractor {

        public LinkVarFe(FeatureNames alphabet) {
            super(alphabet);
        }

        @Override
        public FeatureVector calcFeatureVector(FeExpFamFactor f, VarConfig vc) {
            // No features if one of the values is false.
            for (Var v : vc.getVars()) {
                if (vc.getState(v) == LinkVar.FALSE) {
                    return new FeatureVector();
                }
            }
            // Identity features for the other arcs.
            FeatureVector fv = new FeatureVector();
            if (f instanceof FeTypedFactor) {
                LinkVar lv = (LinkVar) f.getVars().get(0);
                fv.add(alphabet.lookupIndex(lv.getParent() + "_" + lv.getChild()), 1.0);
            } else if (f instanceof GraFeTypedFactor) {
                GraFeTypedFactor ff = (GraFeTypedFactor)f;
                fv.add(alphabet.lookupIndex(ff.g + "_" + ff.p + "_" + ff.c), 1.0);
            } else {
                throw new RuntimeException("unsupported factor type");
            }
            return fv;
        }
        
    }
    
    @Test
    public void testGradByFiniteDiffsAllSemiringsFast() {
          helpGradByFiniteDiffsAllSemirings(false, QLists.getList("a", "b"));
    }
    
    @Ignore("Useful test, but too slow to be included normally.")
    @Test
    public void testGradByFiniteDiffsAllSemirings() {
        helpGradByFiniteDiffsAllSemirings(true, QLists.getList("a"));
        helpGradByFiniteDiffsAllSemirings(true, QLists.getList("a", "b"));
        helpGradByFiniteDiffsAllSemirings(true, QLists.getList("a", "b", "c"));
        helpGradByFiniteDiffsAllSemirings(true, QLists.getList("a", "b", "c", "d"));
        helpGradByFiniteDiffsAllSemirings(false, QLists.getList("a"));
        helpGradByFiniteDiffsAllSemirings(false, QLists.getList("a", "b"));
        helpGradByFiniteDiffsAllSemirings(false, QLists.getList("a", "b", "c"));
        helpGradByFiniteDiffsAllSemirings(false, QLists.getList("a", "b", "c", "d"));
    }

    protected void helpGradByFiniteDiffsAllSemirings(boolean zeroModel, List<String> words) {
        Pair<FactorGraph,FgModel> pair = getO2AllGraFgAndModel(zeroModel, words);
        final FactorGraph fg = pair.get1();
        FgModel model = pair.get2();
        FactorsModule modIn = new FactorsModule(new FgModelIdentity(model), fg, RealAlgebra.getInstance());
        modIn.forward();
        OneToOneFactory<Factors,Beliefs> fact = new OneToOneFactory<Factors,Beliefs>() {
            public Module<Beliefs> getModule(Module<Factors> m1) {
                return new O2AllGraFgInferencer(fg, m1);
            }
        };
        AbstractModuleTest.evalOneToOneByFiniteDiffsAbs(fact, modIn);
    }

}
