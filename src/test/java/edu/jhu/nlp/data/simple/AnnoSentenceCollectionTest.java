package edu.jhu.nlp.data.simple;

import static org.junit.Assert.*;

import java.util.List;

import org.junit.Test;

public class AnnoSentenceCollectionTest {

    @Test
    public void testGetFolds2from7() throws Exception {
        AnnoSentenceCollection sents = getSents(7);
        List<AnnoSentenceCollection> folds = sents.getFolds(2);
        assertEquals(2, folds.size());
        assertEquals(3, folds.get(0).size());
        assertEquals(4, folds.get(1).size());
        assertEquals(sents.get(0), folds.get(0).get(0));
        assertEquals(sents.get(1), folds.get(0).get(1));
        assertEquals(sents.get(2), folds.get(0).get(2));
        assertEquals(sents.get(3), folds.get(1).get(0));
        assertEquals(sents.get(4), folds.get(1).get(1));
        assertEquals(sents.get(5), folds.get(1).get(2));
        assertEquals(sents.get(6), folds.get(1).get(3));
    }
    
    @Test
    public void testGetFolds3from7() throws Exception {
        AnnoSentenceCollection sents = getSents(7);
        List<AnnoSentenceCollection> folds = sents.getFolds(3);
        assertEquals(3, folds.size());
        assertEquals(2, folds.get(0).size());
        assertEquals(2, folds.get(1).size());
        assertEquals(3, folds.get(2).size());
        assertEquals(sents.get(0), folds.get(0).get(0));
        assertEquals(sents.get(1), folds.get(0).get(1));
        assertEquals(sents.get(2), folds.get(1).get(0));
        assertEquals(sents.get(3), folds.get(1).get(1));
        assertEquals(sents.get(4), folds.get(2).get(0));
        assertEquals(sents.get(5), folds.get(2).get(1));
        assertEquals(sents.get(6), folds.get(2).get(2));
    }

    @Test
    public void testGetFolds5from9() throws Exception {
        AnnoSentenceCollection sents = getSents(9);
        List<AnnoSentenceCollection> folds = sents.getFolds(5);
        assertEquals(5, folds.size());
        assertEquals(1, folds.get(0).size());
        assertEquals(1, folds.get(1).size());
        assertEquals(1, folds.get(2).size());
        assertEquals(1, folds.get(3).size());
        assertEquals(5, folds.get(4).size());
    }   

    public AnnoSentenceCollection getSents(int n) {
        AnnoSentenceCollection sents = new AnnoSentenceCollection();
        for (int i=0; i<n; i++) {
            sents.add(AlphabetStoreTest.getAnnoSentenceForRange(i, i+5));
        }
        return sents;
    }
    
}
