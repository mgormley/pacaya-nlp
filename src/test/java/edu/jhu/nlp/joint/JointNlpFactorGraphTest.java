package edu.jhu.nlp.joint;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.junit.Test;

import edu.jhu.nlp.CorpusStatistics;
import edu.jhu.nlp.CorpusStatistics.CorpusStatisticsPrm;
import edu.jhu.nlp.ObsFeTypedFactor;
import edu.jhu.nlp.data.DepEdgeMask;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.data.simple.AnnoSentenceCollection;
import edu.jhu.nlp.joint.JointNlpFactorGraph.JointNlpFactorGraphPrm;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.RoleStructure;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.RoleVar;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.SenseVar;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.SrlFactorTemplate;
import edu.jhu.pacaya.gm.feat.FactorTemplateList;
import edu.jhu.pacaya.gm.feat.ObsFeatureConjoiner;
import edu.jhu.pacaya.gm.feat.ObsFeatureConjoiner.ObsFeatureConjoinerPrm;
import edu.jhu.pacaya.gm.inf.BeliefPropagation;
import edu.jhu.pacaya.gm.inf.BeliefPropagation.BeliefPropagationPrm;
import edu.jhu.pacaya.gm.model.Factor;
import edu.jhu.pacaya.gm.model.FgModel;
import edu.jhu.pacaya.gm.model.Var;
import edu.jhu.pacaya.gm.model.Var.VarType;
import edu.jhu.pacaya.gm.model.VarSet;
import edu.jhu.pacaya.gm.model.VarTensor;
import edu.jhu.pacaya.gm.model.globalfac.LinkVar;
import edu.jhu.pacaya.gm.util.BipartiteGraph;
import edu.jhu.pacaya.util.collections.QLists;
import edu.jhu.prim.Primitives;
import edu.jhu.prim.set.IntHashSet;

/**
 * Unit tests for {@link JointNlpFactorGraph}.
 * @author mgormley
 */
// TODO: This only tests joint dependency parsing and semantic role labeling, but skips relation extraction.
public class JointNlpFactorGraphTest {

    @Test
    public void testNSquaredModel() {
        JointNlpFactorGraphPrm prm = new JointNlpFactorGraphPrm();
        prm.srlPrm.roleStructure = RoleStructure.ALL_PAIRS;
        prm.dpPrm.linkVarType = VarType.LATENT;
        prm.srlPrm.makeUnknownPredRolesLatent = false;
        prm.srlPrm.allowPredArgSelfLoops = false;
        prm.dpPrm.useProjDepTreeFactor = false;
        JointNlpFactorGraph sfg = getJointNlpFg(prm);
        
        LinkVar link = sfg.getLinkVar(1, 2);
        assertNotNull(link);
        assertEquals(1, link.getParent());
        assertEquals(2, link.getChild());

        assertNull(sfg.getLinkVar(1, 1));

        RoleVar role = sfg.getRoleVar(1, 2);
        assertNotNull(role);
        assertEquals(1, role.getParent());
        assertEquals(2, role.getChild());
        
        assertNull(sfg.getRoleVar(1, 1));
        
        List<Var> vars = sfg.getVars();
        assertEquals(9, VarSet.getVarsOfType(vars, VarType.LATENT).size());
        assertEquals(6, VarSet.getVarsOfType(vars, VarType.PREDICTED).size());
        // 6 unary Role, 6 binary Role Link, 9 unary Link.
        assertEquals(6 + 6 + 9, sfg.getNumFactors());
    }

    @Test
    public void testNSquaredModelIsTree() {
        JointNlpFactorGraphPrm prm = new JointNlpFactorGraphPrm();
        prm.srlPrm.roleStructure = RoleStructure.ALL_PAIRS;
        prm.dpPrm.linkVarType = VarType.LATENT;
        prm.srlPrm.makeUnknownPredRolesLatent = false;
        prm.srlPrm.allowPredArgSelfLoops = false;
        prm.dpPrm.useProjDepTreeFactor = false;
        prm.srlPrm.predictSense = true;
        prm.srlPrm.binarySenseRoleFactors = true;
        JointNlpFactorGraph sfg = getJointNlpFg(prm);
                
        BipartiteGraph<Var,Factor> bg = sfg.getBipgraph();
        assertTrue(bg.isAcyclic());
    }
    
    @Test
    public void testPredsGiven() {
        JointNlpFactorGraphPrm prm = new JointNlpFactorGraphPrm();
        prm.srlPrm.roleStructure = RoleStructure.PREDS_GIVEN;
        prm.dpPrm.linkVarType = VarType.LATENT;
        prm.srlPrm.makeUnknownPredRolesLatent = false;
        prm.srlPrm.allowPredArgSelfLoops = false;
        prm.dpPrm.useProjDepTreeFactor = false;
        JointNlpFactorGraph sfg = getJointNlpFg(prm);
        
        LinkVar link = sfg.getLinkVar(-1, 2);
        assertNotNull(link);
        assertEquals(-1, link.getParent());
        assertEquals(2, link.getChild());

        assertNull(sfg.getLinkVar(1, 1));

        RoleVar role = sfg.getRoleVar(1, 2);
        assertNull(role);
        
        assertNull(sfg.getRoleVar(1, 1));
        
        List<Var> vars = sfg.getVars();
        assertEquals(9, VarSet.getVarsOfType(vars, VarType.LATENT).size());
        assertEquals(4, VarSet.getVarsOfType(vars, VarType.PREDICTED).size());
        // 4 unary Role, 4 binary Role Link, 9 unary Link.
        assertEquals(4 + 4 + 9, sfg.getNumFactors());
    }
    
    @Test
    public void testRoleSelfLoops() {
        JointNlpFactorGraphPrm prm = new JointNlpFactorGraphPrm();
        prm.srlPrm.roleStructure = RoleStructure.ALL_PAIRS;
        prm.dpPrm.linkVarType = VarType.LATENT;
        prm.srlPrm.makeUnknownPredRolesLatent = false;
        prm.srlPrm.allowPredArgSelfLoops = true;
        prm.dpPrm.useProjDepTreeFactor = false;
        JointNlpFactorGraph sfg = getJointNlpFg(prm);

        assertNotNull(sfg.getRoleVar(1, 1));
        assertNotNull(sfg.getRoleVar(2, 2));
        
        List<Var> vars = sfg.getVars();
        assertEquals(9, VarSet.getVarsOfType(vars, VarType.LATENT).size());
        assertEquals(9, VarSet.getVarsOfType(vars, VarType.PREDICTED).size());
        // 9 unary Role, 6 binary Role Link, 9 unary Link.
        assertEquals(9 + 6 + 9, sfg.getNumFactors());
    }
    
    @Test
    public void testLinksPredictedRolesLatent() {
        JointNlpFactorGraphPrm prm = new JointNlpFactorGraphPrm();
        prm.srlPrm.roleStructure = RoleStructure.ALL_PAIRS;
        prm.dpPrm.linkVarType = VarType.PREDICTED;
        prm.srlPrm.makeUnknownPredRolesLatent = true;
        prm.srlPrm.allowPredArgSelfLoops = false;
        prm.dpPrm.useProjDepTreeFactor = false;
        JointNlpFactorGraph sfg = getJointNlpFg(prm);
        
        List<Var> vars = sfg.getVars();
        assertEquals(2, VarSet.getVarsOfType(vars, VarType.LATENT).size());
        assertEquals(9 + 4, VarSet.getVarsOfType(vars, VarType.PREDICTED).size());
        // 6 unary Role, 6 binary Role Link, 9 unary Link.
        assertEquals(6 + 6 + 9, sfg.getNumFactors());
    }

    @Test
    public void testUseProjDepTreeFactor() {
        JointNlpFactorGraphPrm prm = new JointNlpFactorGraphPrm();
        prm.srlPrm.roleStructure = RoleStructure.ALL_PAIRS;
        prm.dpPrm.linkVarType = VarType.LATENT;
        prm.srlPrm.makeUnknownPredRolesLatent = false;
        prm.srlPrm.allowPredArgSelfLoops = false;
        prm.dpPrm.useProjDepTreeFactor = true;
        JointNlpFactorGraph sfg = getJointNlpFg(prm);
        
        LinkVar link = sfg.getLinkVar(1, 2);
        assertNotNull(link);
        assertEquals(1, link.getParent());
        assertEquals(2, link.getChild());

        assertNull(sfg.getLinkVar(1, 1));

        RoleVar role = sfg.getRoleVar(1, 2);
        assertNotNull(role);
        assertEquals(1, role.getParent());
        assertEquals(2, role.getChild());
        
        assertNull(sfg.getRoleVar(1, 1));
        
        List<Var> vars = sfg.getVars();
        assertEquals(9, VarSet.getVarsOfType(vars, VarType.LATENT).size());
        assertEquals(6, VarSet.getVarsOfType(vars, VarType.PREDICTED).size());
        // 6 unary Role, 6 binary Role Link, 9 unary Link, 1 global.
        assertEquals(6 + 6 + 9 + 1, sfg.getNumFactors());
    }

    @Test
    public void testPredictSense() {
        JointNlpFactorGraphPrm prm = new JointNlpFactorGraphPrm();
        prm.srlPrm.roleStructure = RoleStructure.PREDS_GIVEN;
        prm.srlPrm.predictSense = true;
        prm.srlPrm.predictPredPos = false;
        JointNlpFactorGraph sfg = getJointNlpFg(prm);
        
        // Assertions about the Sense variables.
        assertNotNull(sfg.getSenseVar(0));
        assertNull(sfg.getSenseVar(1));
        assertNotNull(sfg.getSenseVar(2));
        
        assertEquals(VarType.PREDICTED, sfg.getSenseVar(0).getType());
        assertEquals(VarType.PREDICTED, sfg.getSenseVar(2).getType());
        
        assertEquals(0, sfg.getSenseVar(0).getParent());
        assertEquals(2, sfg.getSenseVar(2).getParent());
        
        assertEquals(QLists.getList("w1.01", "w1.02"), sfg.getSenseVar(0).getStateNames());
        assertEquals(QLists.getList("w3.01", "w3.02"), sfg.getSenseVar(2).getStateNames());
        
        // Assertions about the Sense factors.
        int numSenseFactors = 0;
        for (Factor f : sfg.getFactors()) {
            if (f instanceof ObsFeTypedFactor) {
                ObsFeTypedFactor srlf = (ObsFeTypedFactor) f;
                if (srlf.getFactorType() == SrlFactorTemplate.SENSE_UNARY) {
                    assertEquals(1, srlf.getVars().size());
                    assertTrue(srlf.getVars().iterator().next() instanceof SenseVar);
                    numSenseFactors++;
                }
            }
        }        
        assertEquals(2, numSenseFactors);
    }
    
    @Test
    public void testFirstOrderDepParser() {
        JointNlpFactorGraphPrm prm = new JointNlpFactorGraphPrm();
        prm.includeSrl = false;
        prm.includeDp = true;
        prm.dpPrm.linkVarType = VarType.PREDICTED;
        prm.dpPrm.unaryFactors = true;
        JointNlpFactorGraph sfg = getJointNlpFg(prm);
        
        assertEquals(9, sfg.getFactors().size());
        assertTrue(sfg.getBipgraph().isAcyclic());
    }

    @Test
    public void testSecondOrderDepParser() {
        JointNlpFactorGraphPrm prm = new JointNlpFactorGraphPrm();
        prm.includeSrl = false;
        prm.dpPrm.linkVarType = VarType.PREDICTED;
        prm.dpPrm.unaryFactors = true;
        JointNlpFactorGraph sfg;
        
        // Grandparents only
        prm.dpPrm.grandparentFactors = true;
        prm.dpPrm.arbitrarySiblingFactors = false;
        sfg = getJointNlpFg(prm);        
        assertEquals(9 + 10, sfg.getFactors().size());
        assertFalse(sfg.getBipgraph().isAcyclic());

        // Siblings only 
        prm.dpPrm.grandparentFactors = false;
        prm.dpPrm.arbitrarySiblingFactors = true;
        sfg = getJointNlpFg(prm);        
        assertEquals(9 + 6, sfg.getFactors().size());
        assertFalse(sfg.getBipgraph().isAcyclic());
        
        // Siblings and Grandparents 
        prm.dpPrm.excludeNonprojectiveGrandparents = false;
        prm.dpPrm.grandparentFactors = true;
        prm.dpPrm.arbitrarySiblingFactors = true;
        sfg = getJointNlpFg(prm);     
        assertEquals(9 + 12 + 6, sfg.getFactors().size());
        assertFalse(sfg.getBipgraph().isAcyclic());
    }
    
    @Test
    public void testSecondOrderDepParserPruned() {
        JointNlpFactorGraphPrm prm = new JointNlpFactorGraphPrm();
        prm.includeSrl = false;
        prm.dpPrm.linkVarType = VarType.PREDICTED;
        prm.dpPrm.unaryFactors = true;       
        JointNlpFactorGraph sfg;
        
        // Siblings and Grandparents 
        prm.dpPrm.excludeNonprojectiveGrandparents = false;
        prm.dpPrm.grandparentFactors = true;
        prm.dpPrm.arbitrarySiblingFactors = true;
        prm.dpPrm.pruneEdges = true;
        sfg = getJointNlpFg(prm);     
        assertEquals(11, sfg.getFactors().size());
        System.out.println(sfg.getFactors());
        // This pruned version is a tree.
        assertTrue(sfg.getBipgraph().isAcyclic());
        
        sfg.updateFromModel(new FgModel(1000));
        
        BeliefPropagationPrm bpPrm = new BeliefPropagationPrm();
        BeliefPropagation bp = new BeliefPropagation(sfg, bpPrm);
        bp.run();
        
        // Marginals should yield a left-branching tree.        
        System.out.println("\n\nVariable marginals:\n");
        for (Var v : sfg.getVars()) {
            VarTensor marg = bp.getMarginals(v);
            if (v instanceof LinkVar) {
                LinkVar link = (LinkVar) v;
                if (link.getParent() + 1 == link.getChild()) {
                    // Is left-branching edge.
                    assertTrue(!Primitives.equals(1.0, marg.getValue(LinkVar.FALSE), 1e-13)); 
                    assertTrue(!Primitives.equals(0.0, marg.getValue(LinkVar.TRUE), 1e-13)); 
                } else {
                    // Not left-branching edge.
                    assertEquals(1.0, marg.getValue(LinkVar.FALSE), 1e-13); 
                    assertEquals(0.0, marg.getValue(LinkVar.TRUE), 1e-13); 
                }
            }
            System.out.println(marg);
        }
    }
    
    public static JointNlpFactorGraph getJointNlpFg(JointNlpFactorGraphPrm prm) {
        // --- These won't even be used in these tests ---
        FactorTemplateList fts = new FactorTemplateList();
        ObsFeatureConjoiner ofc = new ObsFeatureConjoiner(new ObsFeatureConjoinerPrm(), fts);
        // ---                                         ---
        Map<String,List<String>> psMap = new HashMap<String,List<String>>() {

            @Override
            public List<String> get(Object predicate) {
                return QLists.getList(predicate + ".01", predicate + ".02");
            }
            
        };
        IntHashSet knownPreds = IntHashSet.fromArray(0, 2);
        List<String> words = QLists.getList("w1", "w2", "w3");        
        // Prune all but a left branching tree.
        DepEdgeMask depEdgeMask = new DepEdgeMask(words.size(), false);
        for (int c=0; c<words.size(); c++) {
            depEdgeMask.setIsKept(c-1, c, true);
        }

        AnnoSentence sent = new AnnoSentence();
        sent.setWords(words);
        sent.setLemmas(words);
        sent.setKnownPreds(knownPreds);
        sent.setDepEdgeMask(depEdgeMask);
        AnnoSentenceCollection sents = new AnnoSentenceCollection(QLists.getList(sent));
        
        CorpusStatistics cs = new CorpusStatistics(new CorpusStatisticsPrm());
        cs.init(sents);
        cs.roleStateNames = QLists.getList("A1", "A2", "A3");
        cs.predSenseListMap = psMap;
        
        prm.srlPrm.srlFePrm.biasOnly = true;
        prm.dpPrm.dpFePrm.biasOnly = true;
        JointNlpFactorGraph fg = new JointNlpFactorGraph(prm, sent, cs, ofc);
        
        return fg;
    }
    
}
