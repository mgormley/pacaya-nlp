package edu.jhu.nlp.features;

import static org.junit.Assert.assertEquals;

import java.util.ArrayList;

import org.junit.Test;

import edu.jhu.nlp.CorpusStatistics;
import edu.jhu.nlp.CorpusStatistics.CorpusStatisticsPrm;
import edu.jhu.nlp.data.conll.CoNLL09Sentence;
import edu.jhu.nlp.data.simple.AlphabetStore;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.data.simple.IntAnnoSentence;
import edu.jhu.nlp.features.TemplateLanguage.FeatTemplate;
import edu.jhu.nlp.features.TemplateLanguage.FeatTemplate0;
import edu.jhu.nlp.features.TemplateLanguage.FeatTemplate3;
import edu.jhu.nlp.features.TemplateLanguage.ListModifier;
import edu.jhu.nlp.features.TemplateLanguage.OtherFeat;
import edu.jhu.nlp.features.TemplateLanguage.PositionList;
import edu.jhu.nlp.features.TemplateLanguage.TokProperty;
import edu.jhu.nlp.words.PrefixAnnotator;
import edu.jhu.pacaya.util.collections.QLists;
import edu.jhu.prim.list.IntArrayList;


public class IntTemplateFeatureExtractorTest {

    @Test
    public void testToFeatIntInt() throws Exception {
        // These collide (surprisingly)
        System.out.println("0x"+Integer.toHexString(5));
        System.out.println("0x"+Integer.toHexString(-579523657));
        System.out.println("0x"+ Long.toHexString(((int)(-579523657) & 0xffffffffl) | ((5 & 0xffffffffl) << 32)));
        System.out.println("0x"+ Long.toHexString(toLong(-579523657, 5)));
        System.out.println("0x"+ Long.toHexString(((long)(-5) & 0xffffffffl) | ((5 & 0xffffffffl) << 32)));
        System.out.println("0x"+ Long.toHexString(toLong(-5, 5)));
        System.out.println("0x"+Long.toHexString(-1808323105l));
//        assertEquals(-1808323105, IntTemplateFeatureExtractor.toFeat(-579523657, 0));
//        assertEquals(-1808323105, IntTemplateFeatureExtractor.toFeat(-579523657, 2));
//        assertEquals(-1808323105, IntTemplateFeatureExtractor.toFeat(-579523657, 5));
//        assertEquals(-1808323105, IntTemplateFeatureExtractor.toFeat(-579523657, 10));
//        assertEquals(-1808323105, IntTemplateFeatureExtractor.toFeat(-579523657, 15));
//        
//        // These are different (as expected)
//        assertEquals(318517609, IntTemplateFeatureExtractor.toFeat(100, 0));
//        assertEquals(-980203759, IntTemplateFeatureExtractor.toFeat(100, 2));
    }
    
    private static final long INT_MAX = 0xffffffffl;
    private static long toLong(int f1, int f2) {
        long l1 =  (((long)f1) & INT_MAX);
        long l2 = (((long)f2) & INT_MAX);
        long feat = l1 | (l2 << 32);
        //long feat =  (((long)f1) & INT_MAX) | ((((long)f2) & INT_MAX) << 32);
        //long feat =  Integer.toUnsignedLong(f1) | ((f2 & INT_MAX) << 32);
        return feat;
    }
    
    @Test
    public void testOtherFeatures() {
        testOtherFeaturesHelper(0, 0, OtherFeat.PATH_LEN, 468451637); // is 1, bin 0
        testOtherFeaturesHelper(4, 2, OtherFeat.PATH_LEN, -346433440); // is 3, bin 2
        testOtherFeaturesHelper(2, 4, OtherFeat.PATH_LEN, -346433440); // is 3, bin 2
        testOtherFeaturesHelper(0, 3, OtherFeat.PATH_LEN, -346433440); // is 4, bin 2
    }

    private void testOtherFeaturesHelper(int pidx, int cidx, OtherFeat f, int expectedVal) {
        FeatTemplate tpl = new FeatTemplate0(f);
        IntTemplateFeatureExtractor extr = getCoNLLSentenceExtractor2();        
        IntArrayList feats = new IntArrayList();
        extr.addFeatures(tpl, LocalObservations.newPidxCidx(pidx, cidx), feats );
        assertEquals(feats.size(), 1);
        assertEquals(expectedVal, feats.get(0));
    }

    private static IntTemplateFeatureExtractor getCoNLLSentenceExtractor1() {
        AnnoSentence sent = CoNLL09Sentence.toAnnoSentence(CoNLL09SentencesForTests.getSpanishConll09Sentence1(), true);
        TemplateFeatureExtractorTest.addFakeAnnos(sent);
        CorpusStatistics cs = new CorpusStatistics(new CorpusStatisticsPrm());
        cs.init(QLists.getList(sent));
        AlphabetStore store = new AlphabetStore(QLists.getList(sent));
        IntAnnoSentence isent = new IntAnnoSentence(sent, store);
        IntTemplateFeatureExtractor extr = new IntTemplateFeatureExtractor(isent, cs);
        return extr;
    }

    private static IntTemplateFeatureExtractor getCoNLLSentenceExtractor2() {
        AnnoSentence sent = CoNLL09Sentence.toAnnoSentence(CoNLL09SentencesForTests.getSpanishConll09Sentence2(), true);
        TemplateFeatureExtractorTest.addFakeAnnos(sent);
        CorpusStatistics cs = new CorpusStatistics(new CorpusStatisticsPrm());
        cs.init(QLists.getList(sent));
        AlphabetStore store = new AlphabetStore(QLists.getList(sent));
        IntAnnoSentence isent = new IntAnnoSentence(sent, store);
        IntTemplateFeatureExtractor extr = new IntTemplateFeatureExtractor(isent, cs);
        return extr;
    }
    
}
