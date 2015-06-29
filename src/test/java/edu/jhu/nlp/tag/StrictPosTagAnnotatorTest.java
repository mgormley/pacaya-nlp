package edu.jhu.nlp.tag;

import static edu.jhu.nlp.tag.StrictPosTagAnnotator.StrictPosTag.CONJ;
import static edu.jhu.nlp.tag.StrictPosTagAnnotator.StrictPosTag.NOUN;
import static edu.jhu.nlp.tag.StrictPosTagAnnotator.StrictPosTag.OTHER;
import static edu.jhu.nlp.tag.StrictPosTagAnnotator.StrictPosTag.PUNC;
import static edu.jhu.nlp.tag.StrictPosTagAnnotator.StrictPosTag.VERB;
import static org.junit.Assert.assertEquals;

import org.junit.Test;

import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.pacaya.util.collections.QLists;

public class StrictPosTagAnnotatorTest {

    @Test
    public void testCposMapping() {
        // Make sentence.
        AnnoSentence sent = new AnnoSentence();
        sent.setWords(QLists.getList("OTHER", "VERB", "NOUN", "PUNC", "CONJ", "OTHER2"));
        sent.setCposTags(QLists.getList("VV-other", "VERB", "NOUN", ".", "CONJ", "NN-OTHER2"));
        // Annotate.
        StrictPosTagAnnotator anno = new StrictPosTagAnnotator();
        anno.annotate(sent);
        // Check.
        assertEquals(QLists.getList(OTHER, VERB, NOUN, PUNC, CONJ, OTHER), sent.getStrictPosTags());
    }

    @Test
    public void testPosMapping() {
        // Make sentence.
        AnnoSentence sent = new AnnoSentence();
        sent.setWords(QLists.getList("OTHER", "VERB", "NOUN", "PUNC", "CONJ", "OTHER2"));
        sent.setPosTags(QLists.getList("OTHER-VV", "VV", "NN", "PUNC", "Conj", "OTHER2-NN"));
        // Annotate.
        StrictPosTagAnnotator anno = new StrictPosTagAnnotator();
        anno.annotate(sent);
        // Check.
        assertEquals(QLists.getList(OTHER, VERB, NOUN, PUNC, CONJ, OTHER), sent.getStrictPosTags());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testNullBehavior() {
        // Make sentence.
        AnnoSentence sent = new AnnoSentence();
        sent.setWords(QLists.getList("OTHER", "VERB", "NOUN", "PUNC", "CONJ", "OTHER2"));
        // Annotate.
        StrictPosTagAnnotator anno = new StrictPosTagAnnotator();
        anno.annotate(sent);
    }

}
