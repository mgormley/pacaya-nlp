package edu.jhu.nlp.features;

import static org.junit.Assert.assertEquals;

import java.util.ArrayList;
import java.util.List;

import org.junit.Test;

import edu.jhu.nlp.CorpusStatistics;
import edu.jhu.nlp.CorpusStatistics.CorpusStatisticsPrm;
import edu.jhu.nlp.data.conll.CoNLL09Sentence;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.data.simple.AnnoSentenceTest;
import edu.jhu.pacaya.util.collections.Lists;

public class FeaturizedTokenTest {

    @Test
    public void testZhaoObjectFeat() {
        CoNLL09Sentence sent = getSpanishConll09Sentence2();
        CorpusStatisticsPrm csPrm = new CorpusStatisticsPrm();
        csPrm.useGoldSyntax = true;
        AnnoSentence simpleSent = sent.toAnnoSentence(csPrm.useGoldSyntax);

        FeaturizedToken zhaoPred = new FeaturizedToken(3, simpleSent);
        FeaturizedToken zhaoArg = new FeaturizedToken(4, simpleSent);
        List<String> predFeat = zhaoPred.getFeat();
        List<String> argFeat = zhaoArg.getFeat();
        ArrayList<String> intendedPredFeats = new ArrayList<String>();
        intendedPredFeats.add("postype=relative");
        intendedPredFeats.add("gen=c");
        intendedPredFeats.add("num=c");
        assertEquals(predFeat,intendedPredFeats);
        ArrayList<String> intendedArgFeats = new ArrayList<String>();
        intendedArgFeats.add("NO_MORPH");
        System.out.println(argFeat);
        assertEquals(argFeat,intendedArgFeats);
    }

    @Test
    public void testZhaoObjectFeat6() {
        CoNLL09Sentence sent = getSpanishConll09Sentence2();
        CorpusStatisticsPrm csPrm = new CorpusStatisticsPrm();
        csPrm.useGoldSyntax = true;
        AnnoSentence simpleSent = sent.toAnnoSentence(csPrm.useGoldSyntax);

        FeaturizedToken zhaoPred = new FeaturizedToken(3, simpleSent);
        FeaturizedToken zhaoArg = new FeaturizedToken(4, simpleSent);
        List<String> predFeat = zhaoPred.getFeat6();
        List<String> argFeat = zhaoArg.getFeat6();
        ArrayList<String> intendedPredFeats = new ArrayList<String>();
        intendedPredFeats.add("postype=relative");
        intendedPredFeats.add("gen=c");
        intendedPredFeats.add("num=c");
        intendedPredFeats.add("NO_MORPH");
        intendedPredFeats.add("NO_MORPH");
        intendedPredFeats.add("NO_MORPH");
        assertEquals(predFeat,intendedPredFeats);
        ArrayList<String> intendedArgFeats = new ArrayList<String>();
        intendedArgFeats.add("NO_MORPH");
        intendedArgFeats.add("NO_MORPH");
        intendedArgFeats.add("NO_MORPH");
        intendedArgFeats.add("NO_MORPH");
        intendedArgFeats.add("NO_MORPH");
        intendedArgFeats.add("NO_MORPH");
        System.out.println(argFeat);
        assertEquals(argFeat,intendedArgFeats);
    }
    
    @Test
    public void testZhaoObjectPos() {
        CoNLL09Sentence sent = getSpanishConll09Sentence2();
        CorpusStatisticsPrm csPrm = new CorpusStatisticsPrm();
        csPrm.useGoldSyntax = true;
        AnnoSentence simpleSent = sent.toAnnoSentence(csPrm.useGoldSyntax);

        FeaturizedToken zhaoPred = new FeaturizedToken(3, simpleSent);
        FeaturizedToken zhaoArg = new FeaturizedToken(4, simpleSent);

        String predPos = zhaoPred.getPos();
        String argPos = zhaoArg.getPos();
        assertEquals(predPos,argPos,"p");
        
        csPrm.useGoldSyntax = false;
        simpleSent = sent.toAnnoSentence(csPrm.useGoldSyntax);

        zhaoPred = new FeaturizedToken(3, simpleSent);
        zhaoArg = new FeaturizedToken(4, simpleSent);
        
        predPos = zhaoPred.getPos();
        argPos = zhaoArg.getPos();
        
        assertEquals(predPos,"p");
        assertEquals(argPos,"WRONG");
    }
    
    @Test
    public void testZhaoObjectParentsChildrenSentence2() {
        CoNLL09Sentence sent = getSpanishConll09Sentence2();
        CorpusStatisticsPrm csPrm = new CorpusStatisticsPrm();
        csPrm.useGoldSyntax = true;
        AnnoSentence simpleSent = sent.toAnnoSentence(csPrm.useGoldSyntax);
        CorpusStatistics cs = new CorpusStatistics(csPrm);
        cs.init(Lists.getList(simpleSent));
        
        FeaturizedToken zhaoObj = new FeaturizedToken(3, simpleSent, cs);
        assertEquals(zhaoObj.getParent(), 5);
        assertEquals(zhaoObj.getChildren(), new ArrayList<Integer>());
        assertEquals(zhaoObj.getFarLeftChild(), -2);
        assertEquals(zhaoObj.getFarLeftChild(), -2);
        assertEquals(zhaoObj.getFarRightChild(), -2);
        assertEquals(zhaoObj.getNearLeftChild(), -2);
        assertEquals(zhaoObj.getNearRightChild(), -2);
        assertEquals(zhaoObj.getHighSupportNoun(), -1);
        assertEquals(zhaoObj.getLowSupportNoun(), -1);
        assertEquals(zhaoObj.getHighSupportVerb(), 1);
        assertEquals(zhaoObj.getLowSupportVerb(), 5);
        ArrayList<Integer> expectedNoFarChildren = new ArrayList<Integer>();
        assertEquals(zhaoObj.getNoFarChildren(), expectedNoFarChildren);
    }

    @Test
    public void testBjorkelundObjectSiblings() {
        CoNLL09Sentence sent = getSpanishConll09Sentence1();
        CorpusStatisticsPrm csPrm = new CorpusStatisticsPrm();
        csPrm.useGoldSyntax = false;
        AnnoSentence simpleSent = sent.toAnnoSentence(csPrm.useGoldSyntax);

        FeaturizedToken b = new FeaturizedToken(3, simpleSent);
        assertEquals(b.getNearRightSibling(), 4);
        assertEquals(b.getNearLeftSibling(), 0);
        b = new FeaturizedToken(0, simpleSent);
        assertEquals(b.getNearLeftSibling(), -1);
        assertEquals(b.getNearRightSibling(), 3);
        sent = getSpanishConll09Sentence2();
        simpleSent = sent.toAnnoSentence(csPrm.useGoldSyntax);

        b = new FeaturizedToken(3, simpleSent);
        // Only true when we're using predicted siblings.
        assertEquals(b.getNearLeftSibling(), -1);
        assertEquals(b.getNearRightSibling(), 7);
        csPrm.useGoldSyntax = true;
        simpleSent = sent.toAnnoSentence(csPrm.useGoldSyntax);

        b = new FeaturizedToken(3, simpleSent);
        assertEquals(b.getNearLeftSibling(), 2);
        assertEquals(b.getNearRightSibling(), 4);
    }
    
    public static CoNLL09Sentence getSpanishConll09Sentence1() {
        return SentFeatureExtractorTest.getSpanishConll09Sentence1(); 
    }
    
    public static CoNLL09Sentence getSpanishConll09Sentence2() {
        return SentFeatureExtractorTest.getSpanishConll09Sentence2(); 
    }
    
    public static CoNLL09Sentence getDogConll09Sentence() {
        return AnnoSentenceTest.getDogConll09Sentence();
    }

}
