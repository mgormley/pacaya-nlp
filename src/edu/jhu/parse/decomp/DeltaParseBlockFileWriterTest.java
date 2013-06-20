package edu.jhu.parse.decomp;

import junit.framework.Assert;

import org.junit.Before;
import org.junit.Test;

import edu.jhu.data.DepTree;
import edu.jhu.data.DepTreebank;
import edu.jhu.data.SentenceCollection;
import edu.jhu.ilp.IlpSolverFactory;
import edu.jhu.ilp.IlpSolverFactory.IlpSolverId;
import edu.jhu.model.Model;
import edu.jhu.model.dmv.DmvModelFactory;
import edu.jhu.model.dmv.RandomDmvModelFactory;
import edu.jhu.parse.DeltaGenerator;
import edu.jhu.parse.FactorDeltaGenerator;
import edu.jhu.parse.FixedIntervalDeltaGenerator;
import edu.jhu.parse.IlpFormulation;
import edu.jhu.parse.IlpDepParserWithDeltas;
import edu.jhu.parse.IlpDepParserWithDeltasTest.MockIlpViterbiParserWithDeltas;
import edu.jhu.util.Prng;


/**
 * TODO: These tests are currently disabled because they are too slow. 
 * @author mgormley
 */
public class DeltaParseBlockFileWriterTest {

    private double lambda = 0.1;

    @Before
    public void setUp() {
        Prng.seed(1234567890);
    }
    
    // DISABLED: @Test
    public void testBranchPriceAndCutPC() {
        SentenceCollection sentences = new SentenceCollection();
        sentences.addSentenceFromString("cat ate mouse");
        sentences.addSentenceFromString("the cat ate the mouse with the hat");
        DmvModelFactory modelFactory = new RandomDmvModelFactory(lambda);
        Model model = modelFactory.getInstance(sentences.getLabelAlphabet());
        double expectedParseWeight;

        DeltaGenerator deltaGen;

        expectedParseWeight = -35.353843;
        deltaGen = new FixedIntervalDeltaGenerator(0.1, 1);
        getParsesMilpBlockPc(model, sentences, IlpFormulation.FLOW_NONPROJ, deltaGen, expectedParseWeight);

        expectedParseWeight = -39.128243;
        deltaGen = new FactorDeltaGenerator(1.1, 2);
        getParsesMilpBlockPc(model, sentences, IlpFormulation.FLOW_PROJ, deltaGen, expectedParseWeight);
    }    
    
    // DISABLED: @Test
    public void testBranchPriceAndCutParsePC() {
        SentenceCollection sentences = new SentenceCollection();
        sentences.addSentenceFromString("cat ate mouse");
        sentences.addSentenceFromString("mouse ate cat");
        //sentences.add(IlpViterbiParserTest.getSentenceFromString("the cat ate the mouse with the hat"));
        DmvModelFactory modelFactory = new RandomDmvModelFactory(lambda);
        Model model = modelFactory.getInstance(sentences.getLabelAlphabet());
        double expectedParseWeight;

        DeltaGenerator deltaGen;

        expectedParseWeight = -35.353843;
        deltaGen = new FixedIntervalDeltaGenerator(0.1, 1);
        getParsesParsePc(model, sentences, IlpFormulation.FLOW_NONPROJ, deltaGen, expectedParseWeight);

        expectedParseWeight = -39.128243;
        deltaGen = new FactorDeltaGenerator(1.1, 2);
        getParsesParsePc(model, sentences, IlpFormulation.FLOW_PROJ, deltaGen, expectedParseWeight);
    }   
    
    // DISABLED: @Test
    public void testBranchAndCutCPM() {
        SentenceCollection sentences = new SentenceCollection();
        sentences.addSentenceFromString("cat ate mouse");
        sentences.addSentenceFromString("the cat ate");
        DmvModelFactory modelFactory = new RandomDmvModelFactory(lambda);
        Model model = modelFactory.getInstance(sentences.getLabelAlphabet());
        double expectedParseWeight;

        DeltaGenerator deltaGen;

        expectedParseWeight = -35.353843;
        deltaGen = new FixedIntervalDeltaGenerator(0.1, 1);
        getParsesCpm(model, sentences, IlpFormulation.FLOW_NONPROJ, deltaGen, expectedParseWeight);

        expectedParseWeight = -39.128243;
        deltaGen = new FactorDeltaGenerator(1.1, 2);
        getParsesCpm(model, sentences, IlpFormulation.FLOW_PROJ, deltaGen, expectedParseWeight);
    }    
    
    public static DepTreebank getParsesMilpBlockPc(Model model, SentenceCollection sentences, IlpFormulation formulation, DeltaGenerator deltaGen, double expectedParseWeight) {
        IlpSolverFactory factory = new IlpSolverFactory(IlpSolverId.DIP_MILPBLOCK_PC, 2, 128);
        factory.setBlockFileWriter(new DeltaParseBlockFileWriter(formulation));
        IlpDepParserWithDeltas parser = new MockIlpViterbiParserWithDeltas(formulation, factory, deltaGen);
        DepTreebank trees = parser.getViterbiParse(sentences, model);
        for (DepTree depTree : trees) {
            System.out.println(depTree);
        }
        Assert.assertEquals(expectedParseWeight, parser.getLastParseWeight(), 1E-13);
        return trees;
    } 
    
    public static DepTreebank getParsesParsePc(Model model, SentenceCollection sentences, IlpFormulation formulation, DeltaGenerator deltaGen, double expectedParseWeight) {
        IlpSolverFactory factory = new IlpSolverFactory(IlpSolverId.DIP_PARSE_PC, 2, 128);
        factory.setBlockFileWriter(new DeltaParseBlockFileWriter(formulation));
        IlpDepParserWithDeltas parser = new MockIlpViterbiParserWithDeltas(formulation, factory, deltaGen);
        DepTreebank trees = parser.getViterbiParse(sentences, model);
        for (DepTree depTree : trees) {
            System.out.println(depTree);
        }
        Assert.assertEquals(expectedParseWeight, parser.getLastParseWeight(), 1E-13);
        return trees;
    }
    
    public static DepTreebank getParsesCpm(Model model, SentenceCollection sentences, IlpFormulation formulation, DeltaGenerator deltaGen, double expectedParseWeight) {
        IlpSolverFactory factory = new IlpSolverFactory(IlpSolverId.DIP_MILPBLOCK_CPM, 2, 128);
        factory.setBlockFileWriter(new DeltaParseBlockFileWriter(formulation));
        IlpDepParserWithDeltas parser = new MockIlpViterbiParserWithDeltas(formulation, factory, deltaGen);
        DepTreebank trees = parser.getViterbiParse(sentences, model);
        for (DepTree depTree : trees) {
            System.out.println(depTree);
        }
        Assert.assertEquals(expectedParseWeight, parser.getLastParseWeight(), 1E-13);
        return trees;
    }
    
}