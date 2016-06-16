package edu.jhu.nlp.features;

import static org.junit.Assert.assertEquals;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;

import org.junit.Test;

import edu.jhu.nlp.CorpusStatistics;
import edu.jhu.nlp.CorpusStatistics.CorpusStatisticsPrm;
import edu.jhu.nlp.data.NerMention;
import edu.jhu.nlp.data.Span;
import edu.jhu.nlp.data.conll.CoNLL09Sentence;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.data.simple.AnnoSentenceTest;
import edu.jhu.nlp.features.TemplateLanguage.EdgeProperty;
import edu.jhu.nlp.features.TemplateLanguage.FeatTemplate;
import edu.jhu.nlp.features.TemplateLanguage.FeatTemplate0;
import edu.jhu.nlp.features.TemplateLanguage.FeatTemplate1;
import edu.jhu.nlp.features.TemplateLanguage.FeatTemplate2;
import edu.jhu.nlp.features.TemplateLanguage.FeatTemplate3;
import edu.jhu.nlp.features.TemplateLanguage.FeatTemplate4;
import edu.jhu.nlp.features.TemplateLanguage.JoinTemplate;
import edu.jhu.nlp.features.TemplateLanguage.ListModifier;
import edu.jhu.nlp.features.TemplateLanguage.OtherFeat;
import edu.jhu.nlp.features.TemplateLanguage.Position;
import edu.jhu.nlp.features.TemplateLanguage.PositionList;
import edu.jhu.nlp.features.TemplateLanguage.PositionModifier;
import edu.jhu.nlp.features.TemplateLanguage.RulePiece;
import edu.jhu.nlp.features.TemplateLanguage.SymbolProperty;
import edu.jhu.nlp.features.TemplateLanguage.TokPropList;
import edu.jhu.nlp.features.TemplateLanguage.TokProperty;
import edu.jhu.nlp.words.PrefixAnnotator;
import edu.jhu.pacaya.parse.cky.Rule;
import edu.jhu.pacaya.util.collections.QLists;
import edu.jhu.prim.bimap.IntObjectBimap;
import edu.jhu.prim.util.math.FastMath;

/**
 * Tests for the template feature extractor.
 * @author mgormley
 */
public class TemplateFeatureExtractorTest {

    @Test
    public void testGetAllUnigrams() {
        extractAllUnigramFeats(0, 3);
    }

    @Test
    public void testGetAllUnigramsWithRootParent() {
        extractAllUnigramFeats(-1, 3);
        extractAllUnigramFeats(3, -1);
    }
    
    private List<String> extractAllUnigramFeats(int pidx, int cidx) {
        int ri = 1;
        int rj = 3;
        int rk = 6;
        int midx = 1;
        Rule rule = getRule("NP", "Det", "N", 0);
        NerMention ne1 = new NerMention(new Span(0,2), "GPE", "Location", "Nom", 1, "UUID-12345");
        NerMention ne2 = new NerMention(new Span(5,9), "PER", "None", "Pro", 5, "UUID-67890");
        LocalObservations local = LocalObservations.getAll(pidx, cidx, midx, rule, ri, rj, rk, ne1, ne2);
        
        AnnoSentence sent = CoNLL09Sentence.toAnnoSentence(AnnoSentenceTest.getDogConll09Sentence(), true);
        addFakeBrownClusters(sent);
        // Add fake coarse POS tags.
        sent.setCposTags(sent.getPosTags());
        
        CorpusStatistics cs = new CorpusStatistics(new CorpusStatisticsPrm());
        cs.init(QLists.getList(sent));
        TemplateFeatureExtractor extr = new TemplateFeatureExtractor(sent, cs);  
        
        List<FeatTemplate> tpls = TemplateSets.getAllUnigramFeatureTemplates();
        ArrayList<String> feats = new ArrayList<String>();
        extr.addFeatures(tpls, local, feats);
        
        for (Object feat : feats) {
            System.out.println(feat);
        }
        return feats;
    }
    

    @Test
    public void testParentPosition() {
        FeatTemplate tpl = new FeatTemplate1(Position.PARENT, PositionModifier.IDENTITY, TokProperty.WORD);
        String expectedFeat = tpl.getName()+"_the";
        getFeatAndAssertEquality(tpl, expectedFeat);
    }

    @Test
    public void testChildPosition() {
        FeatTemplate tpl = new FeatTemplate1(Position.CHILD, PositionModifier.IDENTITY, TokProperty.WORD);
        String expectedFeat = tpl.getName()+"_food";
        getFeatAndAssertEquality(tpl, expectedFeat);
    }
    
    @Test
    public void testPositionModifiers() {
        testPositionModifiersHelper(1, PositionModifier.IDENTITY, "Resultaban");
        testPositionModifiersHelper(1, PositionModifier.LMC, "_");
        testPositionModifiersHelper(1, PositionModifier.LNC, "_");
        testPositionModifiersHelper(1, PositionModifier.RMC, ".");
        testPositionModifiersHelper(1, PositionModifier.RNC, "baratos");
        //
        testPositionModifiersHelper(3, PositionModifier.LNS, "_");
        testPositionModifiersHelper(3, PositionModifier.RNS, "para");
        testPositionModifiersHelper(4, PositionModifier.LNS, "baratos");
        testPositionModifiersHelper(4, PositionModifier.RNS, ".");
        //
        testPositionModifiersHelper(5, PositionModifier.HEAD, "para");
        testPositionModifiersHelper(4, PositionModifier.BEFORE1, "baratos");
        testPositionModifiersHelper(4, PositionModifier.AFTER1, "ser");
        // 
        testPositionModifiersHelper(6, PositionModifier.LOW_SV, "ser");
        testPositionModifiersHelper(6, PositionModifier.HIGH_SV, "Resultaban");
        testPositionModifiersHelper(6, PositionModifier.LOW_SN, "BOS");
        testPositionModifiersHelper(6, PositionModifier.HIGH_SN, "BOS");
        testPositionModifiersHelper(1, PositionModifier.LOW_SV, "Resultaban");
        testPositionModifiersHelper(1, PositionModifier.HIGH_SV, "Resultaban");        
    }

    private void testPositionModifiersHelper(int pidx, PositionModifier mod, String expectedWord) {
        FeatTemplate tpl = new FeatTemplate1(Position.PARENT, mod, TokProperty.WORD);
        TemplateFeatureExtractor extr = getCoNLLSentenceExtractor1();        
        int cidx = -1;
        ArrayList<String> feats = new ArrayList<String>();
        addFeatures(extr, tpl, pidx, cidx, feats);
        for (Object feat : feats) {
            System.out.println(feat);
        }
        assertEquals(feats.size(), 1);
        assertEquals(tpl.getName()+"_"+expectedWord, feats.get(0));
    }
    
    @Test
    public void testTokPropLists() {
        TemplateFeatureExtractor extr = getCoNLLSentenceExtractor1();
        List<String> feats = extr.getTokPropList(TokPropList.EACH_MORPHO, 3);
        String[] expected = new String[]{"postype=qualificative","gen=m","num=p"};
        assertEquals(new HashSet<String>(Arrays.asList(expected)), new HashSet<String>(feats));
    }

    @Test
    public void testTokPropListFeature() {
        FeatTemplate tpl = new FeatTemplate2(Position.PARENT, PositionModifier.IDENTITY, TokPropList.EACH_MORPHO);
        String[] expected = new String[] { tpl.getName()+"_feat1", tpl.getName()+"_feat2" };        
        getFeatsAndAssertEquality(tpl, expected);
    }
    
    @Test
    public void testTokProperties() {
        TemplateFeatureExtractor extr = getCoNLLSentenceExtractor1();
        assertEquals("baratos", extr.getTokProp(TokProperty.WORD, 3));
        assertEquals("barato", extr.getTokProp(TokProperty.LEMMA, 3));
        assertEquals("a", extr.getTokProp(TokProperty.POS, 3));
        assertEquals("postype=qualificative_gen=m_num=p", extr.getTokProp(TokProperty.MORPHO, 3));
        assertEquals("postype=qualificative", extr.getTokProp(TokProperty.MORPHO1, 3));
        assertEquals("gen=m", extr.getTokProp(TokProperty.MORPHO2, 3));
        assertEquals("num=p", extr.getTokProp(TokProperty.MORPHO3, 3));
        assertEquals("11010", extr.getTokProp(TokProperty.BC0, 3));
        assertEquals("1101011", extr.getTokProp(TokProperty.BC1, 3));
        assertEquals("cpred", extr.getTokProp(TokProperty.DEPREL, 3));
        assertEquals("LC", extr.getTokProp(TokProperty.CAPITALIZED, 3));
        // 
        assertEquals("Resultaban", extr.getTokProp(TokProperty.WORD, 1));
        assertEquals("resultaban", extr.getTokProp(TokProperty.LC, 1));
        assertEquals("UC", extr.getTokProp(TokProperty.CAPITALIZED, 1));
        // 
        assertEquals("R", extr.getTokProp(TokProperty.CHPRE1, 1));
        assertEquals("Re", extr.getTokProp(TokProperty.CHPRE2, 1));
        assertEquals("Res", extr.getTokProp(TokProperty.CHPRE3, 1));
        assertEquals("Resu", extr.getTokProp(TokProperty.CHPRE4, 1));
        assertEquals("Resul", extr.getTokProp(TokProperty.CHPRE5, 1));
        assertEquals("taban", extr.getTokProp(TokProperty.CHSUF5, 1));
        assertEquals("n", extr.getTokProp(TokProperty.CHSUF1, 1));
        assertEquals("an", extr.getTokProp(TokProperty.CHSUF2, 1));
        assertEquals("ban", extr.getTokProp(TokProperty.CHSUF3, 1));
        assertEquals("aban", extr.getTokProp(TokProperty.CHSUF4, 1));
        assertEquals("taban", extr.getTokProp(TokProperty.CHSUF5, 1));
    }
    
    @Test
    public void testBosProperties() {
        TemplateFeatureExtractor extr = getCoNLLSentenceExtractor1();
        assertEquals("BOS", extr.getTokProp(TokProperty.WORD, -1));
        assertEquals("BOS", extr.getTokProp(TokProperty.LEMMA, -1));
        assertEquals("BOS", extr.getTokProp(TokProperty.POS, -1));
        assertEquals("BOS", extr.getTokProp(TokProperty.MORPHO, -1));
        assertEquals("BOS", extr.getTokProp(TokProperty.MORPHO1, -1));
        assertEquals("BOS", extr.getTokProp(TokProperty.MORPHO2, -1));
        assertEquals("BOS", extr.getTokProp(TokProperty.MORPHO3, -1));
        assertEquals("BOS", extr.getTokProp(TokProperty.BC0, -1));
        assertEquals("BOS", extr.getTokProp(TokProperty.BC1, -1));
        assertEquals("BOS", extr.getTokProp(TokProperty.DEPREL, -1));
        assertEquals("BOS", extr.getTokProp(TokProperty.LC, -1));
        assertEquals("BOS", extr.getTokProp(TokProperty.CHPRE1, -1));
        assertEquals("BOS", extr.getTokProp(TokProperty.CHPRE2, -1));
        assertEquals("BOS", extr.getTokProp(TokProperty.CHPRE3, -1));
        assertEquals("BOS", extr.getTokProp(TokProperty.CHPRE4, -1));
        assertEquals("BOS", extr.getTokProp(TokProperty.CHPRE5, -1));
        assertEquals("BOS", extr.getTokProp(TokProperty.CHSUF1, -1));
        assertEquals("BOS", extr.getTokProp(TokProperty.CHSUF2, -1));
        assertEquals("BOS", extr.getTokProp(TokProperty.CHSUF3, -1));
        assertEquals("BOS", extr.getTokProp(TokProperty.CHSUF4, -1));
        assertEquals("BOS", extr.getTokProp(TokProperty.CHSUF5, -1));
    }
    
    @Test    
    public void testEosProperties() {
        TemplateFeatureExtractor extr = getCoNLLSentenceExtractor1();
        int n = CoNLL09SentencesForTests.getSpanishConll09Sentence1().size();
        assertEquals("EOS", extr.getTokProp(TokProperty.WORD, n));
        assertEquals("EOS", extr.getTokProp(TokProperty.LEMMA, n));
        assertEquals("EOS", extr.getTokProp(TokProperty.POS, n));
        assertEquals("EOS", extr.getTokProp(TokProperty.MORPHO, n));
        assertEquals("EOS", extr.getTokProp(TokProperty.MORPHO1, n));
        assertEquals("EOS", extr.getTokProp(TokProperty.MORPHO2, n));
        assertEquals("EOS", extr.getTokProp(TokProperty.MORPHO3, n));
        assertEquals("EOS", extr.getTokProp(TokProperty.BC0, n));
        assertEquals("EOS", extr.getTokProp(TokProperty.BC1, n));
        assertEquals("EOS", extr.getTokProp(TokProperty.DEPREL, n));
        assertEquals("EOS", extr.getTokProp(TokProperty.LC, n));
        assertEquals("EOS", extr.getTokProp(TokProperty.CHPRE1, n));
        assertEquals("EOS", extr.getTokProp(TokProperty.CHPRE2, n));
        assertEquals("EOS", extr.getTokProp(TokProperty.CHPRE3, n));
        assertEquals("EOS", extr.getTokProp(TokProperty.CHPRE4, n));
        assertEquals("EOS", extr.getTokProp(TokProperty.CHPRE5, n));
        assertEquals("EOS", extr.getTokProp(TokProperty.CHSUF1, n));
        assertEquals("EOS", extr.getTokProp(TokProperty.CHSUF2, n));
        assertEquals("EOS", extr.getTokProp(TokProperty.CHSUF3, n));
        assertEquals("EOS", extr.getTokProp(TokProperty.CHSUF4, n));
        assertEquals("EOS", extr.getTokProp(TokProperty.CHSUF5, n));
    }
    
    @Test
    public void testTokPropertyAndEdgePropertyNulls1() {        
        FeatTemplate tpl = new FeatTemplate3(PositionList.PATH_P_C, TokProperty.DEPREL, EdgeProperty.DIR, ListModifier.SEQ);
        String expectedFeat = tpl.getName()+"_det_UP_subj_UP_v_DOWN_obj";
        getFeatAndAssertEquality(tpl, expectedFeat);
    }
    
    @Test
    public void testTokPropertyAndEdgePropertyNulls2() {
        FeatTemplate tpl = new FeatTemplate3(PositionList.PATH_P_C, TokProperty.DEPREL, null, ListModifier.SEQ);
        String expectedFeat = tpl.getName()+"_det_subj_v_obj";
        getFeatAndAssertEquality(tpl, expectedFeat);
    }
    
    @Test
    public void testTokPropertyAndEdgePropertyNulls3() {
        FeatTemplate tpl = new FeatTemplate3(PositionList.PATH_P_C, null, EdgeProperty.EDGEREL, ListModifier.SEQ);
        String expectedFeat = tpl.getName()+"_det_subj_obj";
        getFeatAndAssertEquality(tpl, expectedFeat);
    }

    @Test
    public void testListModifiers() {
        {
            FeatTemplate tpl = new FeatTemplate3(PositionList.LINE_P_C, TokProperty.POS, null, ListModifier.SEQ);
            String expectedFeat = tpl.getName()+"_Det_N_V_N";
            getFeatAndAssertEquality(tpl, expectedFeat);
        }
        {
            FeatTemplate tpl = new FeatTemplate3(PositionList.LINE_P_C, TokProperty.POS, null, ListModifier.BAG);
            String expectedFeat = tpl.getName()+"_Det_N_V";
            getFeatAndAssertEquality(tpl, expectedFeat);
        }
        {
            FeatTemplate tpl = new FeatTemplate3(PositionList.LINE_P_C, TokProperty.MORPHO, null, ListModifier.NO_DUP);
            String expectedFeat = tpl.getName()+"_feat1_feat2_feat";
            getFeatAndAssertEquality(tpl, expectedFeat);
        }
    }
    
    @Test
    public void testPositionLists() {
        testPositionListsHelper(2, 3, PositionList.PATH_P_C, "lo_UP_hicieron_DOWN_que", true);
        testPositionListsHelper(2, 4, PositionList.PATH_P_C, "lo_UP_hicieron_DOWN__", true);
        testPositionListsHelper(2, 3, PositionList.PATH_C_LCA, "que_UP_hicieron", true);
        testPositionListsHelper(2, 3, PositionList.PATH_P_LCA, "lo_UP_hicieron", true);
        testPositionListsHelper(2, 3, PositionList.PATH_LCA_ROOT, "hicieron_UP_es_UP_BOS", true);
        
        testPositionListsHelper(2, 3, PositionList.LINE_P_C, "lo_que", false);
        testPositionListsHelper(2, 2, PositionList.LINE_P_C, "lo", false);
        testPositionListsHelper(0, 2, PositionList.LINE_P_C, "Eso_es_lo", false);
        testPositionListsHelper(0, 6, PositionList.LINE_P_C, "Eso_es_lo_que___hicieron_.", false);
        
        // 3 children on the left
        testPositionListsHelper(5, -1, PositionList.CHILDREN_P, "lo_que__", false);
        testPositionListsHelper(5, -1, PositionList.NO_FAR_CHILDREN_P, "que__", false);
        testPositionListsHelper(-1, 5, PositionList.CHILDREN_C, "lo_que__", false);
        testPositionListsHelper(-1, 5, PositionList.NO_FAR_CHILDREN_C, "que__", false);
        // 1 left and 2 right
        testPositionListsHelper(1, -1, PositionList.CHILDREN_P, "Eso_hicieron_.", false);
        testPositionListsHelper(1, -1, PositionList.NO_FAR_CHILDREN_P, "hicieron", false);
        testPositionListsHelper(-1, 1, PositionList.CHILDREN_C, "Eso_hicieron_.", false);
        testPositionListsHelper(-1, 1, PositionList.NO_FAR_CHILDREN_C, "hicieron", false);
        // No children
        testPositionListsHelper(6, -1, PositionList.CHILDREN_P, "", false);
        testPositionListsHelper(6, -1, PositionList.NO_FAR_CHILDREN_P, "", false);
        testPositionListsHelper(-1, 6, PositionList.CHILDREN_C, "", false);
        testPositionListsHelper(-1, 6, PositionList.NO_FAR_CHILDREN_C, "", false);
    }

    private void testPositionListsHelper(int pidx, int cidx, PositionList pl, String expectedVal, boolean includeDir) {
        FeatTemplate tpl = new FeatTemplate3(pl, TokProperty.WORD, includeDir ? EdgeProperty.DIR : null, ListModifier.SEQ);
        TemplateFeatureExtractor extr = getCoNLLSentenceExtractor2();        
        ArrayList<String> feats = new ArrayList<String>();
        addFeatures(extr, tpl, pidx, cidx, feats);
        for (Object feat : feats) {
            System.out.println(feat);
        }
        assertEquals(feats.size(), 1);
        assertEquals(tpl.getName() + "_" + expectedVal, feats.get(0));
    }
    
    @Test
    public void testRuleLocalFeatures() {      
        Rule rule = getRule("S", "NP", "VP", 0);
        testRuleLocalFeaturesHelper(rule, RulePiece.PARENT, SymbolProperty.TAG, "S");
        testRuleLocalFeaturesHelper(rule, RulePiece.LEFT_CHILD, SymbolProperty.TAG, "NP");
        testRuleLocalFeaturesHelper(rule, RulePiece.RIGHT_CHILD, SymbolProperty.TAG, "VP");
    }

    private Rule getRule(String pStr, String lcStr, String rcStr, int type) {
        IntObjectBimap<String> lexAlphabet = new IntObjectBimap<String>();
        IntObjectBimap<String> ntAlphabet = new IntObjectBimap<String>();
        int parent = ntAlphabet.lookupIndex(pStr);        
        int leftChild;
        int rightChild;
        if (rcStr == null && type == Rule.LEXICAL_RULE) {
            leftChild = lexAlphabet.lookupIndex(lcStr);
            rightChild = type;
        }else if (rcStr == null && type == Rule.UNARY_RULE) {
            leftChild = ntAlphabet.lookupIndex(lcStr);
            rightChild = type;        
        } else {
            leftChild = ntAlphabet.lookupIndex(lcStr);
            rightChild = ntAlphabet.lookupIndex(rcStr);
        }
        double score = 1.234;
        Rule rule = new Rule(parent, leftChild, rightChild, score, ntAlphabet, lexAlphabet);
        return rule;
    }
    
    private void testRuleLocalFeaturesHelper(Rule rule, RulePiece piece, SymbolProperty prop, String expectedVal) {
        FeatTemplate tpl = new FeatTemplate4(piece, prop);
        TemplateFeatureExtractor extr = getCoNLLSentenceExtractor2();        
        ArrayList<String> feats = new ArrayList<String>();
        extr.addFeatures(tpl, LocalObservations.newRule(rule), feats);
        for (Object feat : feats) {
            System.out.println(feat);
        }
        assertEquals(feats.size(), 1);
        assertEquals(tpl.getName() + "_" + expectedVal, feats.get(0));
    }

    @Test
    public void testOtherFeatures() {
        testOtherFeaturesHelper(2, 4, OtherFeat.CONTINUITY, "1");
        //
        testOtherFeaturesHelper(2, 4, OtherFeat.DISTANCE, "2");
        testOtherFeaturesHelper(4, 2, OtherFeat.DISTANCE, "2");
        //
        testOtherFeaturesHelper(5, 3, OtherFeat.UNDIR_EDGE, "T");
        testOtherFeaturesHelper(3, 5, OtherFeat.UNDIR_EDGE, "T");
        testOtherFeaturesHelper(1, 2, OtherFeat.UNDIR_EDGE, "F");
        testOtherFeaturesHelper(2, 1, OtherFeat.UNDIR_EDGE, "F");
        testOtherFeaturesHelper(-1, 2, OtherFeat.UNDIR_EDGE, "F");
        //
        testOtherFeaturesHelper(5, 3, OtherFeat.DIR_EDGE, "T");
        testOtherFeaturesHelper(3, 5, OtherFeat.DIR_EDGE, "F");
        testOtherFeaturesHelper(-1, 5, OtherFeat.DIR_EDGE, "F");
        //
        testOtherFeaturesHelper(5, 3, OtherFeat.GENEOLOGY, "PARENT");
        testOtherFeaturesHelper(3, 5, OtherFeat.GENEOLOGY, "CHILD");
        testOtherFeaturesHelper(1, 2, OtherFeat.GENEOLOGY, "ANCESTOR");
        testOtherFeaturesHelper(2, 1, OtherFeat.GENEOLOGY, "DESCENDENT");
        testOtherFeaturesHelper(0, 2, OtherFeat.GENEOLOGY, "COUSIN");
        testOtherFeaturesHelper(2, 0, OtherFeat.GENEOLOGY, "COUSIN");
        testOtherFeaturesHelper(4, 2, OtherFeat.GENEOLOGY, "SIBLING");
        //
        testOtherFeaturesHelper(4, 2, OtherFeat.PATH_LEN, "2"); // is 3
        testOtherFeaturesHelper(2, 4, OtherFeat.PATH_LEN, "2"); // is 3
        testOtherFeaturesHelper(0, 3, OtherFeat.PATH_LEN, "2"); // is 4
        testOtherFeaturesHelper(0, 0, OtherFeat.PATH_LEN, "0"); // is 1
        // 
        testOtherFeaturesHelper(0, 0, OtherFeat.SENT_LEN, "5"); // is 7
        //
        testOtherFeaturesHelper(4, 2, OtherFeat.RELATIVE, "AFTER");
        testOtherFeaturesHelper(2, 2, OtherFeat.RELATIVE, "ON");
        testOtherFeaturesHelper(2, 4, OtherFeat.RELATIVE, "BEFORE");
    }

    private void testOtherFeaturesHelper(int pidx, int cidx, OtherFeat f, String expectedVal) {
        FeatTemplate tpl = new FeatTemplate0(f);
        TemplateFeatureExtractor extr = getCoNLLSentenceExtractor2();        
        ArrayList<String> feats = new ArrayList<String>();
        addFeatures(extr, tpl, pidx, cidx, feats);
        for (Object feat : feats) {
            System.out.println(feat);
        }
        assertEquals(feats.size(), 1);
        assertEquals(tpl.getName() + "_" + expectedVal, feats.get(0));
    }
    
    private void testOtherFeaturesHelper2(int pidx, int cidx, FeatTemplate tpl, String... expectedFeats) {
        TemplateFeatureExtractor extr = getCoNLLSentenceExtractor2();        
        ArrayList<String> feats = new ArrayList<String>();
        addFeatures(extr, tpl, pidx, cidx, feats);
        for (Object feat : feats) {
            System.out.println(feat);
        }
        assertEquals(new HashSet<String>(Arrays.asList(expectedFeats)), new HashSet<Object>(feats));
    }
    
    @Test
    public void testPathGramsFeature() {      
        FeatTemplate tpl = new FeatTemplate0(OtherFeat.PATH_GRAMS);

        String[] expectedPathGrams = new String[] { "pathGrams_the", "pathGrams_Det", "pathGrams_dog",
                "pathGrams_N", "pathGrams_ate", "pathGrams_V", "pathGrams_food", "pathGrams_N",
                "pathGrams_the_dog", "pathGrams_Det_dog", "pathGrams_the_N", "pathGrams_Det_N",
                "pathGrams_dog_ate", "pathGrams_N_ate", "pathGrams_dog_V", "pathGrams_N_V", "pathGrams_ate_food",
                "pathGrams_V_food", "pathGrams_ate_N", "pathGrams_V_N", "pathGrams_the_dog_ate",
                "pathGrams_Det_dog_ate", "pathGrams_the_N_ate", "pathGrams_Det_N_ate", "pathGrams_the_dog_V",
                "pathGrams_Det_dog_V", "pathGrams_the_N_V", "pathGrams_Det_N_V", "pathGrams_dog_ate_food",
                "pathGrams_N_ate_food", "pathGrams_dog_V_food", "pathGrams_N_V_food", "pathGrams_dog_ate_N",
                "pathGrams_N_ate_N", "pathGrams_dog_V_N", "pathGrams_N_V_N", };
        
        getFeatsAndAssertEquality(tpl, expectedPathGrams);
    }
    
    @Test
    public void testBtwnPosFeature() {  
        FeatTemplate tpl = new FeatTemplate3(PositionList.BTWN_P_C, TokProperty.POS, null, ListModifier.UNIGRAM);
        testOtherFeaturesHelper2(0, 3, tpl, tpl.getName()+"_d", tpl.getName()+"_v");
        testOtherFeaturesHelper2(3, 0, tpl, tpl.getName()+"_d", tpl.getName()+"_v");
        testOtherFeaturesHelper2(1, 3, tpl, tpl.getName()+"_d");
        testOtherFeaturesHelper2(3, 1, tpl, tpl.getName()+"_d");
        testOtherFeaturesHelper2(0, 2, tpl, tpl.getName()+"_v");

        testOtherFeaturesHelper2(0, 1, tpl);
        testOtherFeaturesHelper2(1, 0, tpl);
        testOtherFeaturesHelper2(1, 1, tpl);
    }
    
    @Test
    public void testBigramFeature() {   
        {
            // Single feature.
            FeatTemplate tpl1 = new FeatTemplate1(Position.PARENT, PositionModifier.IDENTITY, TokProperty.MORPHO);
            String expected1 = tpl1.getName()+"_feat1_feat2";
            FeatTemplate tpl2 = new FeatTemplate3(PositionList.LINE_P_C, TokProperty.POS, null, ListModifier.SEQ);
            String expected2 = tpl2.getName()+"_Det_N_V_N";
            FeatTemplate tpl = new JoinTemplate(tpl1, tpl2);
            getFeatAndAssertEquality(tpl, expected1 + "_" + expected2);
        }
        {
            // Multiple features.
            FeatTemplate tpl1 = new FeatTemplate2(Position.PARENT, PositionModifier.IDENTITY, TokPropList.EACH_MORPHO);
            String[] expected1 = new String[] { tpl1.getName()+"_feat1",
                    tpl1.getName()+"_feat2" };
            FeatTemplate tpl2 = new FeatTemplate3(PositionList.LINE_P_C, TokProperty.POS, null, ListModifier.SEQ);
            String expected2 = tpl2.getName()+"_Det_N_V_N";
            FeatTemplate tpl = new JoinTemplate(tpl1, tpl2);
            String[] expected = new String[] { expected1[0] + "_" + expected2, expected1[1] + "_" + expected2 };
            getFeatsAndAssertEquality(tpl, expected);
        }
    }
    
    @Test
    public void testTrigramFeature() {   
        // Multiple features.
        FeatTemplate tpl1 = new FeatTemplate2(Position.PARENT, PositionModifier.IDENTITY, TokPropList.EACH_MORPHO);
        String[] expected1 = new String[] { tpl1.getName()+"_feat1",
                tpl1.getName()+"_feat2" };
        FeatTemplate tpl2 = new FeatTemplate3(PositionList.LINE_P_C, TokProperty.POS, null, ListModifier.SEQ);
        String expected2 = tpl2.getName()+"_Det_N_V_N";
        FeatTemplate tpl3 = new FeatTemplate1(Position.PARENT, PositionModifier.IDENTITY, TokProperty.MORPHO);
        String expected3 = tpl3.getName()+"_feat1_feat2";
        FeatTemplate tpl = new JoinTemplate(tpl1, tpl2, tpl3);
        String[] expected = new String[] { expected1[0] + "_" + expected2 + "_" + expected3, 
                expected1[1] + "_" + expected2  + "_" + expected3};
        getFeatsAndAssertEquality(tpl, expected);
    }
    
    // Single feature.
    private void getFeatAndAssertEquality(FeatTemplate tpl, String expectedFeat) {
        TemplateFeatureExtractor extr = getDogSentenceExtractor();  
        
        int pidx = 0;
        int cidx = 3;
        ArrayList<String> feats = new ArrayList<String>();
        addFeatures(extr, tpl, pidx, cidx, feats);

        for (Object feat : feats) {
            System.out.println(feat);
        }
        assertEquals(feats.size(), 1);
        assertEquals(expectedFeat, feats.get(0));
    }

    // Multiple features
    private void getFeatsAndAssertEquality(FeatTemplate tpl, String[] expectedPathGrams) {
        TemplateFeatureExtractor extr = getDogSentenceExtractor();  
        int pidx = 0;
        int cidx = 3;
        ArrayList<String> feats = new ArrayList<String>();
        addFeatures(extr, tpl, pidx, cidx, feats);

        for (Object feat : feats) {
            System.out.println(feat);
        }
        assertEquals(new HashSet<String>(Arrays.asList(expectedPathGrams)), new HashSet<Object>(feats));
    }

    private static TemplateFeatureExtractor getDogSentenceExtractor() {
        AnnoSentence sent = CoNLL09Sentence.toAnnoSentence(AnnoSentenceTest.getDogConll09Sentence(), true);
        CorpusStatistics cs = new CorpusStatistics(new CorpusStatisticsPrm());
        cs.init(QLists.getList(sent));
        TemplateFeatureExtractor extr = new TemplateFeatureExtractor(sent, cs);
        return extr;
    }

    private static TemplateFeatureExtractor getCoNLLSentenceExtractor1() {
        AnnoSentence sent = CoNLL09Sentence.toAnnoSentence(CoNLL09SentencesForTests.getSpanishConll09Sentence1(), true);
        addFakeBrownClusters(sent);
        PrefixAnnotator.addPrefixes(sent);
        CorpusStatistics cs = new CorpusStatistics(new CorpusStatisticsPrm());
        cs.init(QLists.getList(sent));
        TemplateFeatureExtractor extr = new TemplateFeatureExtractor(sent, cs);
        return extr;
    }

    private static TemplateFeatureExtractor getCoNLLSentenceExtractor2() {
        AnnoSentence sent = CoNLL09Sentence.toAnnoSentence(CoNLL09SentencesForTests.getSpanishConll09Sentence2(), true);
        addFakeBrownClusters(sent);
        CorpusStatistics cs = new CorpusStatistics(new CorpusStatisticsPrm());
        cs.init(QLists.getList(sent));
        TemplateFeatureExtractor extr = new TemplateFeatureExtractor(sent, cs);
        return extr;
    }
    
    public static void addFakeBrownClusters(AnnoSentence sent) {
        ArrayList<String> clusters = new ArrayList<String>();
        for (int i=0; i<sent.size(); i++) {
            clusters.add(FastMath.mod(i*7, 2) + "10101" + FastMath.mod(i*39, 2));
        }
        sent.setClusters(clusters);
    }
    

    
    /** Adds features for a single feature template. (The parent index and child index are the only local observations.) */
    // TODO: Remove this method when convenient.
    private static void addFeatures(TemplateFeatureExtractor extr, FeatTemplate tpl, int pidx, int cidx, List<String> feats) {
        extr.addFeatures(tpl, LocalObservations.newPidxCidx(pidx, cidx), feats);
    }

}
