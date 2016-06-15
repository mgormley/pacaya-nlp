package edu.jhu.nlp.srl;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;

import java.util.List;

import org.junit.Test;

import edu.jhu.nlp.CorpusStatistics;
import edu.jhu.nlp.CorpusStatistics.CorpusStatisticsPrm;
import edu.jhu.nlp.data.DepGraph;
import edu.jhu.nlp.data.conll.SrlGraph;
import edu.jhu.nlp.data.simple.AlphabetStore;
import edu.jhu.nlp.data.simple.AlphabetStoreTest;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.data.simple.AnnoSentenceCollection;
import edu.jhu.nlp.data.simple.IntAnnoSentence;
import edu.jhu.nlp.joint.JointNlpFactorGraph;
import edu.jhu.nlp.joint.JointNlpFactorGraph.JointNlpFactorGraphPrm;
import edu.jhu.nlp.joint.JointNlpFactorGraphTest;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.RoleStructure;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.RoleVar;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.SenseVar;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.SrlFactorGraphBuilderPrm;
import edu.jhu.pacaya.gm.feat.FactorTemplateList;
import edu.jhu.pacaya.gm.feat.ObsFeatureConjoiner;
import edu.jhu.pacaya.gm.feat.ObsFeatureConjoiner.ObsFeatureConjoinerPrm;
import edu.jhu.pacaya.gm.model.Factor;
import edu.jhu.pacaya.gm.model.FactorGraph;
import edu.jhu.pacaya.gm.model.Var;
import edu.jhu.pacaya.gm.model.Var.VarType;
import edu.jhu.pacaya.gm.model.VarConfig;
import edu.jhu.pacaya.gm.model.VarSet;
import edu.jhu.pacaya.util.collections.QLists;
import edu.jhu.prim.set.IntHashSet;

public class SrlFactorGraphBuilderTest {

    @Test
    public void testPredictPredPos() {
        JointNlpFactorGraphPrm prm = new JointNlpFactorGraphPrm();
        prm.includeDp = false;
        prm.srlPrm.roleStructure = RoleStructure.ALL_PAIRS;
        prm.srlPrm.makeUnknownPredRolesLatent = false;
        prm.srlPrm.allowPredArgSelfLoops = false;
        prm.srlPrm.predictPredPos = true;
        prm.srlPrm.binarySenseRoleFactors = true;
        JointNlpFactorGraph sfg = JointNlpFactorGraphTest.getJointNlpFg(prm);
                
        System.out.print("Vars: ");
        for (Var var : sfg.getVars()) {
            System.out.print(var.getName() + " ");
        }
        System.out.println();
        
        System.out.print("Factors: ");
        for (Factor f : sfg.getFactors()) {
            System.out.print(f.getVars() + " ");
        }
        System.out.println();
        
        assertNotNull(sfg.getRoleVar(1, 2));
        assertNull(sfg.getRoleVar(1, 1));
        
        List<Var> vars = sfg.getVars();
        assertEquals(0, VarSet.getVarsOfType(vars, VarType.LATENT).size());
        assertEquals(9, VarSet.getVarsOfType(vars, VarType.PREDICTED).size());
        // 6 unary Role, 6 binary Role Sense, 3 sense unary.
        assertEquals(6 + 6 + 3, sfg.getNumFactors());
    }
    
    @Test
    public void testPredictPredSenseAndPredPos() {
        JointNlpFactorGraphPrm prm = new JointNlpFactorGraphPrm();
        prm.includeDp = false;
        prm.srlPrm.roleStructure = RoleStructure.ALL_PAIRS;
        prm.srlPrm.makeUnknownPredRolesLatent = false;
        prm.srlPrm.allowPredArgSelfLoops = false;
        prm.srlPrm.predictSense = true;
        prm.srlPrm.predictPredPos = true;
        prm.srlPrm.binarySenseRoleFactors = true;
        JointNlpFactorGraph sfg = JointNlpFactorGraphTest.getJointNlpFg(prm);
                
        System.out.print("Vars: ");
        for (Var var : sfg.getVars()) {
            System.out.print(var.getName() + " ");
        }
        System.out.println();
        
        System.out.print("Factors: ");
        for (Factor f : sfg.getFactors()) {
            System.out.print(f.getVars() + " ");
        }
        System.out.println();
        
        assertNotNull(sfg.getRoleVar(1, 2));
        assertNull(sfg.getRoleVar(1, 1));
        
        List<Var> vars = sfg.getVars();
        assertEquals(0, VarSet.getVarsOfType(vars, VarType.LATENT).size());
        assertEquals(9, VarSet.getVarsOfType(vars, VarType.PREDICTED).size());
        // 6 unary Role, 6 binary Role Sense, 3 sense unary.
        assertEquals(6 + 6 + 3, sfg.getNumFactors());
    }
    
    /* ------------------------------- Decode --------------------------------- */
    
    
    @Test
    public void testGetSrlGraph() {
        int n = 3;
        SrlFactorGraphBuilder srlBuilder = getSrlBuilder(n);
        for (int i=0; i<n; i++) {
            System.out.println(srlBuilder.getSenseVar(i));
        }
        
        VarConfig vc = new VarConfig();
        vc.put(srlBuilder.getSenseVar(0), "sense0");
        vc.put(srlBuilder.getRoleVar(0, 0), "_");
        vc.put(srlBuilder.getRoleVar(0, 1), "role0_1");
        vc.put(srlBuilder.getRoleVar(0, 2), "role0_2");
        // Self-loop
        vc.put(srlBuilder.getSenseVar(2), "sense2");
        vc.put(srlBuilder.getRoleVar(2, 0), "_");
        vc.put(srlBuilder.getRoleVar(2, 1), "_");
        vc.put(srlBuilder.getRoleVar(2, 2), "role2_2");
        
        DepGraph g = srlBuilder.getSrlGraphFromMbrVarConfig(vc);
        System.out.println(g);
        assertEquals("sense0", g.get(-1, 0));
        assertEquals(null, g.get(0, 0));
        assertEquals("role0_1", g.get(0, 1));
        assertEquals("role0_2", g.get(0, 2));
        assertEquals("sense2", g.get(-1, 2));
        assertEquals(null, g.get(2, 0));
        assertEquals(null, g.get(2, 1));
        assertEquals("role2_2", g.get(2, 2));        
    }

    @Test(expected=RuntimeException.class)
    public void testGetSrlGraphNoVars() {
        int n = 3;
        SrlFactorGraphBuilder srlBuilder = getSrlBuilder(n);
        VarConfig vc = new VarConfig();
        DepGraph g = srlBuilder.getSrlGraphFromMbrVarConfig(vc);
        // This code will never be reached:
        System.out.println(g);
        for (int i = -1; i < n; i++) {
            for (int j = 0; j < n; j++) {
                assertEquals(null, g.get(i, j));
            }
        }
    }
    
    private static SrlFactorGraphBuilder getSrlBuilder(int n) {
        AnnoSentence sent = getAnnoSentenceWithSrl(n);
        AnnoSentenceCollection sents = new AnnoSentenceCollection(QLists.getList(sent));
        AlphabetStore store = new AlphabetStore(sents);
        IntAnnoSentence isent = new IntAnnoSentence(sent, store);
        CorpusStatistics cs = new CorpusStatistics(new CorpusStatisticsPrm());
        cs.init(sents);
        FactorTemplateList fts = new FactorTemplateList();
        ObsFeatureConjoiner ofc = new ObsFeatureConjoiner(new ObsFeatureConjoinerPrm(), fts);
        FactorGraph fg = new FactorGraph();

        SrlFactorGraphBuilderPrm prm = new SrlFactorGraphBuilderPrm();
        prm.predictSense = true;
        prm.predictPredPos = false;
        prm.roleStructure = RoleStructure.PREDS_GIVEN;
        prm.allowPredArgSelfLoops = true;
        SrlFactorGraphBuilder srlBuilder = new SrlFactorGraphBuilder(prm);
        srlBuilder.build(isent, cs, ofc, fg);
        return srlBuilder;
    }

    private static AnnoSentence getAnnoSentenceWithSrl(int n) {
        AnnoSentence sent = AlphabetStoreTest.getAnnoSentenceForRange(0, n);
        // Srl graph
        //
        // Every other token is a predicate.
        DepGraph srlGraph = new DepGraph(n);
        IntHashSet knownPreds = new IntHashSet();
        for (int i = -1; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i == -1 && j % 2 == 0) {
                    knownPreds.add(j);
                    srlGraph.set(i, j, "sense" + j);
                }
                if (i != -1 && i % 2 == 0) {
                    srlGraph.set(i, j, "role" + i + "_" + j);
                }
            }
        }
        sent.setKnownPreds(knownPreds);
        sent.setSrlGraph(srlGraph);
        return sent;
    }
        
}
