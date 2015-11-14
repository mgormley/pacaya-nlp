package edu.jhu.nlp.data.simple;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.junit.Test;

import edu.jhu.nlp.features.TemplateLanguage.AT;
import edu.jhu.nlp.tag.StrictPosTagAnnotator.StrictPosTag;
import edu.jhu.pacaya.util.collections.QLists;

public class AlphabetStoreTest {

    public static final int NUM_TOKENS = 100 + AlphabetStore.NUM_SPECIAL_TOKS; 
    public static final String TOK_UNK_STR = AlphabetStore.TOK_UNK_STR;
    public static final int FIRST_TOK_ID = AlphabetStore.NUM_SPECIAL_TOKS;
    
    @Test
    public void testAlphabetStoreNoCutoff() {
        AnnoSentenceCollection sents = getSents(false);
        AlphabetStore store = new AlphabetStore(sents);
        
        // Check alphabet sizes.
        assertEquals(NUM_TOKENS, store.words.size());
        assertEquals(NUM_TOKENS, store.lcWords.size());
        assertEquals(18, store.prefixes.size()); // many fewer than NUM_TOKENS=104 
        assertEquals(414, store.suffixes.size()); // many more than NUM_TOKENS=104
        assertEquals(NUM_TOKENS, store.lemmas.size());
        assertEquals(NUM_TOKENS, store.posTags.size());
        assertEquals(NUM_TOKENS, store.cposTags.size());
        assertEquals(NUM_TOKENS, store.clusters.size());
        assertEquals(9, store.clusterPrefixes.size());
        assertEquals(NUM_TOKENS, store.feats.size());
        assertEquals(NUM_TOKENS, store.deprels.size());
        
        
        // Check alphabet contents.
        assertEquals(TOK_UNK_STR, store.words.lookupObject(0));
        assertEquals("Word"+0, store.words.lookupObject(FIRST_TOK_ID));
        assertEquals(FIRST_TOK_ID, store.getWordIdx("Word"+0));

        assertEquals(TOK_UNK_STR, store.lcWords.lookupObject(0));
        assertEquals("word"+0, store.lcWords.lookupObject(FIRST_TOK_ID)); // lowercase of Word0
        assertEquals(FIRST_TOK_ID, store.getLcWordIdx("word"+0)); // lowercase of Word0
        
        assertEquals(TOK_UNK_STR, store.prefixes.lookupObject(0));
        assertEquals("W", store.prefixes.lookupObject(FIRST_TOK_ID)); // prefix of word0
        assertEquals(FIRST_TOK_ID, store.getPrefixIdx("W")); // prefix of word0
        
        assertEquals(TOK_UNK_STR, store.suffixes.lookupObject(0));
        assertEquals("0", store.suffixes.lookupObject(FIRST_TOK_ID)); // suffix of word0
        assertEquals(FIRST_TOK_ID, store.getSuffixIdx("0")); // suffix of word0
        
        assertEquals(TOK_UNK_STR, store.lemmas.lookupObject(0));
        assertEquals("lemma"+0, store.lemmas.lookupObject(FIRST_TOK_ID));
        assertEquals(FIRST_TOK_ID, store.getLemmaIdx("lemma"+0));

        assertEquals(TOK_UNK_STR, store.posTags.lookupObject(0));
        assertEquals("pos"+0, store.posTags.lookupObject(FIRST_TOK_ID));
        assertEquals(FIRST_TOK_ID, store.getPosTagIdx("pos"+0));

        assertEquals(TOK_UNK_STR, store.cposTags.lookupObject(0));
        assertEquals("cpos"+0, store.cposTags.lookupObject(FIRST_TOK_ID));
        assertEquals(FIRST_TOK_ID, store.getCposTagIdx("cpos"+0));

        assertEquals(TOK_UNK_STR, store.clusters.lookupObject(0));
        assertEquals("cluster"+0, store.clusters.lookupObject(FIRST_TOK_ID));
        assertEquals(FIRST_TOK_ID, store.getClusterIdx("cluster"+0));
        
        assertEquals(TOK_UNK_STR, store.clusterPrefixes.lookupObject(0));
        assertEquals("c", store.clusterPrefixes.lookupObject(FIRST_TOK_ID)); // prefix of cluster0
        assertEquals(FIRST_TOK_ID, store.getClusterPrefixIdx("c")); // prefix of cluster0

        assertEquals(TOK_UNK_STR, store.feats.lookupObject(0));
        assertEquals("feat"+0, store.feats.lookupObject(FIRST_TOK_ID));
        assertEquals(FIRST_TOK_ID, store.getFeatIdx("feat"+0));

        assertEquals(TOK_UNK_STR, store.deprels.lookupObject(0));
        assertEquals("deprel"+0, store.deprels.lookupObject(FIRST_TOK_ID));
        assertEquals(FIRST_TOK_ID, store.getDeprelIdx("deprel"+0));
    }

    @Test
    public void testRemovedAt() {
        AnnoSentenceCollection orig = getSents(false);
        for (AT at : AT.values()) {
            AnnoSentenceCollection sents = orig.getWithAtsRemoved(QLists.getList(at));
            AlphabetStore store = new AlphabetStore(sents);
            
            // Check alphabet sizes.
            assertEquals(at == AT.WORD ? FIRST_TOK_ID : NUM_TOKENS, store.words.size());
            if (at == AT.WORD) {
                assertEquals(FIRST_TOK_ID, store.lcWords.size());
                assertEquals(FIRST_TOK_ID, store.prefixes.size());
                assertEquals(FIRST_TOK_ID, store.suffixes.size());
            } else {
                assertTrue(FIRST_TOK_ID < store.lcWords.size() && store.lcWords.size() <= NUM_TOKENS);
                assertTrue(FIRST_TOK_ID < store.prefixes.size());
                assertTrue(FIRST_TOK_ID < store.suffixes.size());
            }
            //assertEquals(at == AT.PREFIX ? FIRST_TOK_ID : NUM_TOKENS, store.prefixes.size());
            assertEquals(at == AT.LEMMA ? FIRST_TOK_ID : NUM_TOKENS, store.lemmas.size());
            assertEquals(at == AT.POS ? FIRST_TOK_ID : NUM_TOKENS, store.posTags.size());
            assertEquals(at == AT.CPOS ? FIRST_TOK_ID : NUM_TOKENS, store.cposTags.size());
            assertEquals(at == AT.BROWN ? FIRST_TOK_ID : NUM_TOKENS, store.clusters.size());
            if (at == AT.BROWN){
                assertEquals(FIRST_TOK_ID, store.clusterPrefixes.size());
            } else {
                assertTrue(FIRST_TOK_ID < store.clusterPrefixes.size());
            }
            assertEquals(at == AT.MORPHO ? FIRST_TOK_ID : NUM_TOKENS, store.feats.size());
            assertEquals(at == AT.DEPREL ? FIRST_TOK_ID : NUM_TOKENS, store.deprels.size());
        }
    }
    
    /**
     * This tests that if there are TOO MANY types for each of the various alphabets, we will impose
     * a count cutoff which effectively maps all the tokens in the fourth sentence from
     * getSents(true) to UNK.
     */
    @Test
    public void testAlphabetStoreWithCutoff() {
        AnnoSentenceCollection sents = getSents(true);
        AlphabetStore store = new AlphabetStore(sents);
        assertEquals(NUM_TOKENS, store.words.size());
        assertEquals(NUM_TOKENS, store.lcWords.size());
        assertEquals(18, store.prefixes.size());
        assertEquals(11424, store.suffixes.size());
        assertEquals(NUM_TOKENS, store.lemmas.size());
        assertEquals(NUM_TOKENS, store.posTags.size());
        assertEquals(NUM_TOKENS, store.cposTags.size());
        assertEquals(NUM_TOKENS, store.clusters.size());
        assertEquals(9, store.clusterPrefixes.size());
        assertEquals(NUM_TOKENS, store.feats.size());
        assertEquals(NUM_TOKENS, store.deprels.size());
    }
    
    @Test
    public void testWordCounts() {
        {
            AnnoSentenceCollection sents = getSents(true);
            AlphabetStore store = new AlphabetStore(sents);
            // Observed.
            assertEquals(3, store.getWordTypeCount(store.getWordIdx("Word"+0)));
            assertEquals(3, store.getWordTypeCount(store.getWordIdx("Word"+1)));
            assertEquals(3, store.getWordTypeCount(store.getWordIdx("Word"+10)));
            // Mapped to UNK.
            assertEquals(65445, store.getWordTypeCount(store.getWordIdx("Word"+101)));
            assertEquals(65445, store.getWordTypeCount(store.getWordIdx("Word"+1000)));
        }
        {
            AnnoSentenceCollection sents = getSents(false);
            AlphabetStore store = new AlphabetStore(sents);
            // Observed.
            assertEquals(3, store.getWordTypeCount(store.getWordIdx("Word"+0)));
            assertEquals(3, store.getWordTypeCount(store.getWordIdx("Word"+1)));
            assertEquals(3, store.getWordTypeCount(store.getWordIdx("Word"+10)));
            // Mapped to UNK.
            assertEquals(0, store.getWordTypeCount(store.getWordIdx("Word"+101)));
            assertEquals(0, store.getWordTypeCount(store.getWordIdx("Word"+1000)));
        }
    }

    @Test
    public void testWordTopNCutoff() {
        {
            AnnoSentenceCollection sents = getWordOnlySents(1000);
            AlphabetStore store = new AlphabetStore(sents);
            assertEquals(1003, store.words.size());
            assertEquals(199, store.getWordTopNCutoff());
        }{
            AnnoSentenceCollection sents = getWordOnlySents(10);
            AlphabetStore store = new AlphabetStore(sents);
            assertEquals(13, store.words.size());
            assertEquals(1, store.getWordTopNCutoff());
        }
    }

    protected AnnoSentenceCollection getWordOnlySents(int numSents) {
        AnnoSentenceCollection sents = new AnnoSentenceCollection();
        // Add three tokens for word<i> for i in [0,..,99].
        for (int i=0; i<numSents; i++) {
            AnnoSentence s = new AnnoSentence();
            s.setWords(new ArrayList<>());
            for (int j=0; j<i; j++) {
                s.getWords().add("Word"+j);
            }
            sents.add(s);
        }
        return sents;
    }
    
    /**
     * Gets a list of sentences. If <code>includeExtras</code> is false, we return only three
     * sentences of length 100 tokens. This will include 100 unique types for each of the
     * annotations. If <code>includeExtras</code> is true, we add a fourth sentence with many
     * additional types up to the maximum size of a short plus one.
     */
    public static AnnoSentenceCollection getSents(boolean includeExtras) {
        AnnoSentenceCollection sents = new AnnoSentenceCollection();
        // Add three tokens for word<i> for i in [0,..,99].
        for (int j=0; j<3; j++) {
            for (int i=0; i<100; i++) {
                AnnoSentence s = new AnnoSentence();
                s.setWords(QLists.getList("Word"+i));
                s.setPrefixes(QLists.getList("prefix"+i));
                s.setLemmas(QLists.getList("lemma"+i));
                s.setPosTags(QLists.getList("pos"+i));
                s.setCposTags(QLists.getList("cpos"+i));
                s.setClusters(QLists.getList("cluster"+i));
                s.setFeats(QLists.getList(QLists.getList("feat"+i)));
                s.setDeprels(QLists.getList("deprel"+i));
                sents.add(s);
            }
        }
        
        if (includeExtras) {
            // Add one token for word<i> for i in [100,..., 65545].
            int start=100;
            int end = 0xffff+10;
            AnnoSentence s = getAnnoSentenceForRange(start, end);            
            sents.add(s);
        }
        
        return sents;
    }

    public static AnnoSentence getAnnoSentenceForRange(int start, int end) {
        int len = end - start;
        AnnoSentence s = new AnnoSentence();
        s.setWords(getList("Word"+start));
        s.setPrefixes(getList("prefix"+start));
        s.setLemmas(getList("lemma"+start));
        s.setPosTags(getList("pos"+start));
        s.setCposTags(getList("cpos"+start));
        s.setClusters(getList("cluster"+start));
        s.setDeprels(getList("deprel"+start));
        s.setFeats(getList(getList("feat"+start)));    
        s.setStrictPosTags(getList(StrictPosTag.values()[start%StrictPosTag.values().length]));
        
        for (int i=start+1; i<end; i++) {
            s.getWords().add("Word"+i);
            s.getPrefixes().add("prefix"+i);
            s.getLemmas().add("lemma"+i);
            s.getPosTags().add("pos"+i);
            s.getCposTags().add("cpos"+i);
            s.getClusters().add("cluster"+i);
            s.getDeprels().add("deprel"+i);
            s.getFeats().add(getList("feat"+i));
            s.getStrictPosTags().add(StrictPosTag.values()[i%StrictPosTag.values().length]);
        }
        return s;
    }
    
    @Test
    public void testStopGrowth() {
        AlphabetStore store = new AlphabetStore(new AnnoSentenceCollection());
        store.startGrowth();
        assertEquals(true, store.words.isGrowing());
        assertEquals(true, store.lcWords.isGrowing());
        assertEquals(true, store.prefixes.isGrowing());
        assertEquals(true, store.lemmas.isGrowing());
        assertEquals(true, store.posTags.isGrowing());
        assertEquals(true, store.cposTags.isGrowing());
        assertEquals(true, store.clusters.isGrowing());
        assertEquals(true, store.feats.isGrowing());
        assertEquals(true, store.deprels.isGrowing());
        store.stopGrowth();
        assertEquals(false, store.words.isGrowing());
        assertEquals(false, store.lcWords.isGrowing());
        assertEquals(false, store.prefixes.isGrowing());
        assertEquals(false, store.lemmas.isGrowing());
        assertEquals(false, store.posTags.isGrowing());
        assertEquals(false, store.cposTags.isGrowing());
        assertEquals(false, store.clusters.isGrowing());
        assertEquals(false, store.feats.isGrowing());
        assertEquals(false, store.deprels.isGrowing());        
    }
    

    @SafeVarargs
    public static <T> List<T> getList(T... args) {
        ArrayList<T> a = new ArrayList<>();
        a.addAll(Arrays.asList(args));
        return a;
    }    

}
