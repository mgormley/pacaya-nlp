package edu.jhu.nlp.fcm;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.List;

import org.junit.Test;

import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.embed.Embeddings;
import edu.jhu.pacaya.autodiff.ModuleTestUtils;
import edu.jhu.pacaya.autodiff.Tensor;
import edu.jhu.pacaya.gm.feat.FeatureVector;
import edu.jhu.pacaya.gm.model.FactorsModule;
import edu.jhu.pacaya.gm.model.FgModel;
import edu.jhu.pacaya.gm.model.FgModelIdentity;
import edu.jhu.pacaya.gm.model.Var;
import edu.jhu.pacaya.gm.model.Var.VarType;
import edu.jhu.pacaya.gm.model.VarSet;
import edu.jhu.pacaya.gm.model.VarTensor;
import edu.jhu.pacaya.util.FeatureNames;
import edu.jhu.pacaya.util.collections.QLists;
import edu.jhu.pacaya.util.semiring.Algebra;
import edu.jhu.pacaya.util.semiring.LogSemiring;
import edu.jhu.pacaya.util.semiring.RealAlgebra;
import edu.jhu.prim.bimap.IntObjectBimap;
import edu.jhu.prim.tuple.Pair;


public class FcmModuleTest {

    @Test
    public void testSimple1() {
        Algebra s = LogSemiring.getInstance();
        Pair<FgModelIdentity,FcmModule> pair = getFcm1(s);
        FgModelIdentity id1 = pair.get1();
        FcmModule fcm = pair.get2();
        
        fcm.forward();
        VarTensor y = fcm.getOutput();
        System.out.println(y);
        assertEquals(3, y.size());
        assertEquals(0.0998138, y.getValue(0), 1e-4);
        assertEquals(0.0213886, y.getValue(1), 1e-4);
        assertEquals(0.0126206, y.getValue(2), 1e-4);
        
        VarTensor yAdj = fcm.getOutputAdj();
        yAdj.fill(fcm.getAlgebra().fromReal(5));
        
        fcm.backward();
        FgModel grad = id1.getOutputAdj().getModel();
        System.out.println(grad);
        assertEquals(0, grad.getParams().get(0), 1e-1);
        assertEquals(0.1493196192412, grad.getParams().get(1), 1e-1);
        assertEquals(0.2874402670393, grad.getParams().get(2), 1e-1);
        assertEquals(0.1381206477981, grad.getParams().get(3), 1e-1);
        assertEquals(0.2732056769633, grad.getParams().get(10), 1e-1);
        assertEquals(0.1245876796231, grad.getParams().get(20), 1e-1);
        assertEquals(0.2567501458544, grad.getParams().get(30), 1e-1);
        assertEquals(2.616507072526, grad.getParams().get(42), 1e-1);
    }
    
    @Test
    public void testGradByFiniteDiffs1() {
        Algebra s = RealAlgebra.getInstance();
        Pair<FgModelIdentity,FcmModule> pair = getFcm1(s);
        FgModelIdentity id1 = pair.get1();
        FcmModule fcm = pair.get2();
        ModuleTestUtils.assertGradientCorrectByFd(fcm, 1e-5, 1e-8);
    }
    
    public static Pair<FgModelIdentity,FcmModule> getFcm1(Algebra s) {
        // Model - 1 offset, 4*3*3 tensor, 2*3 embeddings.
        FgModel model = new FgModel(1 + 4*3*3 + 2*3);
        model.fill(0.0);
        for (int i=0; i<model.getNumParams(); i++) {
            model.getParams().set(i, 1.0 / i);
        }
        FgModelIdentity id1 = new FgModelIdentity(model);
        // Features
        FeatureNames alphabet = new FeatureNames();
        FeatureVector f1 = new FeatureVector();
        f1.add(alphabet.lookupIndex("f-a"), 1.0);
        f1.add(alphabet.lookupIndex("f-b"), 1.0);
        FeatureVector f2 = new FeatureVector();
        f2.add(alphabet.lookupIndex("f-b"), 1.0);
        f2.add(alphabet.lookupIndex("f-c"), 1.0);
        f2.add(alphabet.lookupIndex("f-d"), 1.0);
        List<FeatureVector> feats = QLists.getList(f1, f2);
        // Var
        VarSet vars = new VarSet(new Var(VarType.PREDICTED, 3, "y", QLists.getList("y-a", "y-b", "y-c")));
        // Sentence
        AnnoSentence sent = new AnnoSentence();
        sent.setWords(QLists.getList("w-a", "w-b"));
        // Embeddings
        IntObjectBimap<String> map = new IntObjectBimap<>();
        for (String w : sent.getWords()) {
            map.lookupIndex(w);
        }
        map.stopGrowth();
        Tensor e = new Tensor(RealAlgebra.getInstance(), 2, 3);
        e.fill(2); // ignored
        Embeddings embeds = new Embeddings(e, map);
        
        FcmModule fcm = new FcmModule(id1, s, feats, alphabet, vars, sent, embeds, 1, true, null);
        return new Pair<>(id1, fcm);
    }
    

    @Test
    public void testSimple2() {
        Algebra s = LogSemiring.getInstance();
        Pair<FgModelIdentity,FcmModule> pair = getFcm2(s);
        FgModelIdentity id1 = pair.get1();
        FcmModule fcm = pair.get2();
        
        fcm.forward();
        VarTensor y = fcm.getOutput();
        System.out.println(y);
        assertEquals(2, y.size());
        assertEquals(27, y.getValue(0), 1e-4);
        assertEquals(59, y.getValue(1), 1e-4);
        
        VarTensor yAdj = fcm.getOutputAdj();
        yAdj.fill(fcm.getAlgebra().fromReal(5));
        
        fcm.backward();
        FgModel grad = id1.getOutputAdj().getModel();
        System.out.println(grad);
        assertEquals(0, grad.getParams().get(0), 1e-1);
        for (int i=1; i<grad.getNumParams(); i++) {
            assertTrue(grad.getParams().get(i) > 100000);
        }
    }

    @Test
    public void testGradByFiniteDiffs2() {
        Algebra s = RealAlgebra.getInstance();
        Pair<FgModelIdentity,FcmModule> pair = getFcm2(s);
        FgModelIdentity id1 = pair.get1();
        FcmModule fcm = pair.get2();
        ModuleTestUtils.assertGradientCorrectByFd(fcm, 1e-5, 1e-8);
    }

    public static Pair<FgModelIdentity,FcmModule> getFcm2(Algebra s) {
        // Model - 1 offset, 2*2*1 tensor, 2*1 embeddings.
        FgModel model = new FgModel(1 + 2*2*1 + 2*1);
        model.fill(0.0);
        for (int i=0; i<model.getNumParams(); i++) {
            model.getParams().set(i, i);
        }
        FgModelIdentity id1 = new FgModelIdentity(model);
        // Features
        FeatureNames alphabet = new FeatureNames();
        FeatureVector f1 = new FeatureVector();
        f1.add(alphabet.lookupIndex("f-a"), 1.0);
        f1.add(alphabet.lookupIndex("f-b"), 1.0);
        FeatureVector f2 = new FeatureVector();
        f2.add(alphabet.lookupIndex("f-b"), 1.0);
        List<FeatureVector> feats = QLists.getList(f1, f2);
        // Var
        VarSet vars = new VarSet(new Var(VarType.PREDICTED, 2, "y", QLists.getList("y-a", "y-b")));
        // Sentence
        AnnoSentence sent = new AnnoSentence();
        sent.setWords(QLists.getList("w-a", "w-b"));
        // Embeddings
        IntObjectBimap<String> map = new IntObjectBimap<>();
        for (String w : sent.getWords()) {
            map.lookupIndex(w);
        }
        map.stopGrowth();
        Tensor e = new Tensor(RealAlgebra.getInstance(), 2, 1);
        e.fill(2); // ignored
        Embeddings embeds = new Embeddings(e, map);
        
        FcmModule fcm = new FcmModule(id1, s, feats, alphabet, vars, sent, embeds, 1, true, null);
        return new Pair<>(id1, fcm);
    }
    
}
