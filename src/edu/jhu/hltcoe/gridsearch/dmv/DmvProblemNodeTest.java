package edu.jhu.hltcoe.gridsearch.dmv;

import static junit.framework.Assert.assertEquals;

import java.io.File;
import java.util.List;

import org.apache.log4j.BasicConfigurator;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import edu.jhu.hltcoe.data.SentenceCollection;
import edu.jhu.hltcoe.gridsearch.ProblemNode;
import edu.jhu.hltcoe.gridsearch.cpt.BasicCptBoundsDeltaFactory;
import edu.jhu.hltcoe.gridsearch.cpt.CptBounds;
import edu.jhu.hltcoe.gridsearch.cpt.CptBoundsDeltaFactory;
import edu.jhu.hltcoe.gridsearch.cpt.MidpointVarSplitter;
import edu.jhu.hltcoe.gridsearch.cpt.RandomVariableSelector;
import edu.jhu.hltcoe.gridsearch.cpt.VariableSelector;
import edu.jhu.hltcoe.gridsearch.cpt.VariableSplitter;
import edu.jhu.hltcoe.gridsearch.cpt.CptBoundsDelta.Type;
import edu.jhu.hltcoe.gridsearch.cpt.MidpointVarSplitter.MidpointChoice;
import edu.jhu.hltcoe.gridsearch.dmv.DmvDantzigWolfeRelaxation.CutCountComputer;
import edu.jhu.hltcoe.train.DmvTrainCorpus;
import edu.jhu.hltcoe.util.Prng;


public class DmvProblemNodeTest {

    static {
        BasicConfigurator.configure();
        //Logger.getRootLogger().setLevel(Level.TRACE);
    }

    @Before
    public void setUp() {
        DmvProblemNode.clearActiveNode();
        Prng.seed(1234567890);
    }
    
    @Test
    public void testBranch() {
        SentenceCollection sentences = new SentenceCollection();
        sentences.addSentenceFromString("N V");
//        sentences.addSentenceFromString("N V P");
//        sentences.addSentenceFromString("N V N N N");
//        sentences.addSentenceFromString("N V P N");
        DmvTrainCorpus corpus = new DmvTrainCorpus(sentences);

        DmvRelaxation relax = new DmvDantzigWolfeRelaxation(new File("."), 100, new CutCountComputer());

        VariableSelector varSelector = new RandomVariableSelector(true);
        VariableSplitter varSplitter = new MidpointVarSplitter(MidpointChoice.HALF_PROB);
        CptBoundsDeltaFactory brancher = new BasicCptBoundsDeltaFactory(varSelector, varSplitter);
        DmvProblemNode node = new DmvProblemNode(corpus, brancher, relax);
        List<ProblemNode> children = node.branch();
        assertEquals(2, children.size());
        DmvProblemNode c1 = (DmvProblemNode)children.get(0);
        DmvProblemNode c2 = (DmvProblemNode)children.get(1);
        
        assertEquals(0, node.getId()); 
        assertEquals(1, c1.getId()); 
        assertEquals(2, c2.getId()); 
        
        assertEquals(0, node.getDepth()); 
        assertEquals(1, c1.getDepth());
                
        IndexedDmvModel idm = new IndexedDmvModel(corpus);
        double[][][] bounds = new double[idm.getNumConds()][][];
        for (int c=0; c<idm.getNumConds(); c++) {
            bounds[c] = new double[idm.getNumParams(c)][2];
            for (int m=0; m<idm.getNumParams(c); m++) {
                CptBounds b = node.getBounds();
                bounds[c][m][0] = b.getLb(Type.PARAM, c, m);
                bounds[c][m][1] = b.getUb(Type.PARAM, c, m);
            }
        }
        
        checkedSetActive(c1, node);
        for (int c=0; c<idm.getNumConds(); c++) {
            CptBounds b = c1.getBounds();                
             for (int m=0; m<idm.getNumParams(c); m++) {
                Assert.assertTrue(bounds[c][m][0] <= b.getLb(Type.PARAM, c, m));
                Assert.assertTrue(bounds[c][m][1] >= b.getUb(Type.PARAM, c, m));
            }
        }
        if (c1.getRelaxation() instanceof DmvDantzigWolfeRelaxation) {
            assertEquals(Math.log(0.5), c1.getBounds().getLb(Type.PARAM, 2, 1), 1e-7);
        } else {
            assertEquals(Math.log(0.5), c1.getBounds().getLb(Type.PARAM, 2, 1), 1e-7);
        }
        
        DmvProblemNode c3 = (DmvProblemNode)c1.branch().get(0);
        
        checkedSetActive(c3, c1);
        
        checkedSetActive(c2, c3);
        if (c2.getRelaxation() instanceof DmvDantzigWolfeRelaxation) {
            assertEquals(Math.log(0.5), c2.getBounds().getUb(Type.PARAM, 2, 1), 1e-7);
        } else {
            assertEquals(Math.log(0.5), c2.getBounds().getUb(Type.PARAM, 2, 1), 1e-7);
        }
        DmvProblemNode c4 = (DmvProblemNode)c2.branch().get(1);
        checkedSetActive(c4, c2);
        for (int c=0; c<idm.getNumConds(); c++) {
            for (int m=0; m<idm.getNumParams(c); m++) {
                CptBounds b = c4.getBounds();                
                Assert.assertTrue(bounds[c][m][0] <= b.getLb(Type.PARAM, c, m));
                Assert.assertTrue(bounds[c][m][1] >= b.getUb(Type.PARAM, c, m));
            }
        }

        checkedSetActive(c3, c4);
        
        DmvProblemNode prev = c3;
        for (int i=0; i<10; i++) {
            DmvProblemNode cur = (DmvProblemNode)prev.branch().get(0);
            checkedSetActive(cur, prev);
            prev = cur;
        }
        prev.end();
    }

    private void checkedSetActive(DmvProblemNode nextNode, DmvProblemNode prevNode) {
        if (nextNode.getParent() != prevNode) {
            nextNode.setAsActiveNode();
        } else {
            double prevBound = prevNode.getOptimisticBound();
            nextNode.setAsActiveNode();
            double nextBound = nextNode.getOptimisticBound();
            System.out.println("parent: " + prevBound + " child: " + nextBound);
            Assert.assertTrue(prevBound >= nextBound);
        }
    }
    
}
