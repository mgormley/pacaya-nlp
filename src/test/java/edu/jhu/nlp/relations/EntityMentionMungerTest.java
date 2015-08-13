package edu.jhu.nlp.relations;

import static org.junit.Assert.*;

import org.junit.Test;

import edu.jhu.nlp.data.NerMention;
import edu.jhu.nlp.data.NerMentions;
import edu.jhu.nlp.data.Span;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.pacaya.util.collections.QLists;


public class EntityMentionMungerTest {

    @Test
    public void testAnnotate() {
        AnnoSentence sent = new AnnoSentence();
        sent.setWords(QLists.getList("a", "b", "c", "d", "e"));
        NerMention ne1 = new NerMention(new Span(1, 2), "type1", "subtype1", "phraseType1", 1, "id1");
        NerMention ne2 = new NerMention(new Span(3, 4), "type2", "subtype2", "phraseType2", 3, "id2");
        NerMentions mentions = new NerMentions(sent.size(), QLists.getList(ne1, ne2));
        sent.setNamedEntities(mentions);
        
        EntityMentionMunger munger = new EntityMentionMunger();
        munger.annotate(sent);
        
        assertEquals(QLists.getList("a", "b-ne", "c", "d-ne", "e"), sent.getWords());
    }
    
}
