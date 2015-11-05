package edu.jhu.nlp.data.simple;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import edu.jhu.nlp.tag.StrictPosTagAnnotator;
import edu.jhu.pacaya.util.collections.QLists;

public class IntAnnoSentenceTest {
    
    @Test    
    public void testKnownTypes() {
        AnnoSentenceCollection sents = AlphabetStoreTest.getSents(false);
        AlphabetStore store = new AlphabetStore(sents);
        
        // Test known words.
        int i = 0;
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
        
        IntAnnoSentence isent = new IntAnnoSentence(s, store);
        assertEquals((short) store.getWordIdx("Word"+0), isent.getWord(0));
        assertEquals((short) store.getLcWordIdx("word"+0), isent.getLcWord(0));
        assertEquals((short) store.getLemmaIdx("lemma"+0), isent.getLemma(0));
        assertEquals((byte) store.getPosTagIdx("pos"+0), isent.getPosTag(0));
        assertEquals((byte) store.getCposTagIdx("cpos"+0), isent.getCposTag(0));
        assertEquals((short) store.getClusterIdx("cluster"+0), isent.getCluster(0));
        assertEquals((short) store.getFeatIdx("feat"+0), isent.getFeats(0).get(0));
        assertEquals((byte) store.getDeprelIdx("deprel"+0), isent.getDeprel(0));

        // Prefixes / Suffixes
        assertEquals((short) store.getPrefixIdx("W"), isent.getPrefix(0,1));
        assertEquals((short) store.getPrefixIdx("Wo"), isent.getPrefix(0,2));
        assertEquals((short) store.getPrefixIdx("Wor"), isent.getPrefix(0,3));
        assertEquals((short) store.getSuffixIdx("0"), isent.getSuffix(0,1));
        assertEquals((short) store.getSuffixIdx("d0"), isent.getSuffix(0,2));
        assertEquals((short) store.getSuffixIdx("rd0"), isent.getSuffix(0,3));
        assertEquals((short) store.getClusterPrefixIdx("c"), isent.getClusterPrefix(0,1));
        assertEquals((short) store.getClusterPrefixIdx("cl"), isent.getClusterPrefix(0,2));
        assertEquals((short) store.getClusterPrefixIdx("clu"), isent.getClusterPrefix(0,3));
    }
    
    // If this is failing, it's likely because the alphabet growth wasn't stopped.
    @Test    
    public void testUnknownTypes() {
        AnnoSentenceCollection sents = AlphabetStoreTest.getSents(false);
        AlphabetStore store = new AlphabetStore(sents);
        
        // Test unknown words.
        String i = "-unseen-suffix";
        AnnoSentence s = new AnnoSentence();
        s.setWords(QLists.getList("unknown-word"+i));
        s.setPrefixes(QLists.getList("prefix"+i));
        s.setLemmas(QLists.getList("lemma"+i));
        s.setPosTags(QLists.getList("pos"+i));
        s.setCposTags(QLists.getList("cpos"+i));
        s.setClusters(QLists.getList("unknown-cluster"+i));
        s.setFeats(QLists.getList(QLists.getList("feat"+i)));
        s.setDeprels(QLists.getList("deprel"+i));
        sents.add(s);
        
        IntAnnoSentence isent = new IntAnnoSentence(s, store);
        assertEquals((short) AlphabetStore.TOK_UNK_INT, isent.getWord(0));
        assertEquals((short) AlphabetStore.TOK_UNK_INT, isent.getLcWord(0));
        assertEquals((short) AlphabetStore.TOK_UNK_INT, isent.getLemma(0));
        assertEquals((byte) AlphabetStore.TOK_UNK_INT, isent.getPosTag(0));
        assertEquals((byte) AlphabetStore.TOK_UNK_INT, isent.getCposTag(0));
        assertEquals((short) AlphabetStore.TOK_UNK_INT, isent.getCluster(0));
        assertEquals((short) AlphabetStore.TOK_UNK_INT, isent.getPrefix(0,1));
        assertEquals((short) AlphabetStore.TOK_UNK_INT, isent.getPrefix(0,2));        
        assertEquals((short) AlphabetStore.TOK_UNK_INT, isent.getFeats(0).get(0));
        assertEquals((byte) AlphabetStore.TOK_UNK_INT, isent.getDeprel(0));

        // Prefixes / Suffixes
        assertEquals((short) AlphabetStore.TOK_UNK_INT, isent.getPrefix(0,1));
        assertEquals((short) AlphabetStore.TOK_UNK_INT, isent.getPrefix(0,2));
        assertEquals((short) AlphabetStore.TOK_UNK_INT, isent.getPrefix(0,3));        
        assertEquals((short) AlphabetStore.TOK_UNK_INT, isent.getSuffix(0,1));
        assertEquals((short) AlphabetStore.TOK_UNK_INT, isent.getSuffix(0,2));
        assertEquals((short) AlphabetStore.TOK_UNK_INT, isent.getSuffix(0,3));
        assertEquals((short) AlphabetStore.TOK_UNK_INT, isent.getClusterPrefix(0,1));
        assertEquals((short) AlphabetStore.TOK_UNK_INT, isent.getClusterPrefix(0,2));
        assertEquals((short) AlphabetStore.TOK_UNK_INT, isent.getClusterPrefix(0,3));
    }
    
    @Test
    public void testNumInBetween() {
        AnnoSentence sent = new AnnoSentence();
        sent.setWords(QLists.getList(   "0",     "1",    "2",    "3", "4",    "5",    "6",    "7"));
        sent.setCposTags(QLists.getList("VERB", "VERB", "NOUN", ".", "CONJ", "NOUN", "VERB", "VERB"));
        AnnoSentenceCollection sents = new AnnoSentenceCollection();
        sents.add(sent);
        AlphabetStore store = new AlphabetStore(sents);
        StrictPosTagAnnotator anno = new StrictPosTagAnnotator();
        anno.annotate(sents);
        
        IntAnnoSentence isent = new IntAnnoSentence(sent, store);
        assertEquals(0, isent.getNumVerbsInBetween(1, 1));
        assertEquals(1, isent.getNumVerbsInBetween(0, 2));
        assertEquals(2, isent.getNumVerbsInBetween(0, 7));
        assertEquals(3, isent.getNumVerbsInBetween(0, 8));
        assertEquals(4, isent.getNumVerbsInBetween(-1, 8));
    }

}
