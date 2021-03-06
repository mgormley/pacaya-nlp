package edu.jhu.nlp.data.simple;

import static org.junit.Assert.assertArrayEquals;

import java.util.ArrayList;
import java.util.List;

import org.junit.Test;

import edu.jhu.nlp.data.conll.CoNLL09Sentence;
import edu.jhu.nlp.data.conll.CoNLL09Token;
import edu.jhu.pacaya.util.collections.QLists;

public class AnnoSentenceTest {

    @Test
    public void testGetParentsAndUseGoldSyntax() {
        CoNLL09Sentence sent = AnnoSentenceTest.getDogConll09Sentence();
        {
            // Test with gold syntax.
            boolean useGoldSyntax = true;
            AnnoSentence simpleSent = sent.toAnnoSentence(useGoldSyntax);
            assertArrayEquals(new int[] { 1, 2, -1, 2 }, simpleSent.getParents());
        }
        {
            // Test without gold syntax.
            boolean useGoldSyntax = false;
            AnnoSentence simpleSent = sent.toAnnoSentence(useGoldSyntax);
            assertArrayEquals(new int[] { 2, 0, -1, 2 }, simpleSent.getParents());
        }
    }

    public static CoNLL09Sentence getDogConll09Sentence() {
        List<CoNLL09Token> tokens = new ArrayList<CoNLL09Token>();
        //tokens.add(new CoNLL09Token(id, form, lemma, plemma, pos, ppos, feat, pfeat, head, phead, deprel, pdeprel, fillpred, pred, apreds));
        tokens.add(new CoNLL09Token(1, "the", "_", "GoldDet", "Det", "_", QLists.getList("feat1","feat2"), QLists.getList("feat1","feat2") , 2, 3, "det", "_", false, "_", QLists.getList("_")));
        tokens.add(new CoNLL09Token(2, "dog", "_", "GoldN", "N", "_", QLists.getList("feat"), QLists.getList("feat") , 3, 1, "subj", "_", false, "_", QLists.getList("arg0")));
        tokens.add(new CoNLL09Token(3, "ate", "_", "GoldV", "V", "_", QLists.getList("feat"), QLists.getList("feat") , 0, 0, "v", "_", true, "ate.1", QLists.getList("_")));
        tokens.add(new CoNLL09Token(4, "food", "_", "GoldN", "N", "_", QLists.getList("feat"), QLists.getList("feat") , 3, 3, "obj", "_", false, "_", QLists.getList("arg1")));
        CoNLL09Sentence sent = new CoNLL09Sentence(tokens);
        
        return sent;
    }
    
    @Test
    public void testSetSrlGraphToNull() {
        // Check that we can set it to null without an error.
        AnnoSentence sent = new AnnoSentence();
        sent.setSrlGraph(null);
    }
    
}
