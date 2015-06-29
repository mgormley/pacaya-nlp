package edu.jhu.nlp.data.semeval;

import java.util.ArrayList;
import java.util.List;

import edu.jhu.nlp.data.NerMention;
import edu.jhu.nlp.data.NerMentions;
import edu.jhu.nlp.data.RelationMention;
import edu.jhu.nlp.data.RelationMentions;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.pacaya.util.collections.QLists;
import edu.jhu.prim.tuple.Pair;

public class SemEval2010Sentence {

    public String id;
    public List<String> words;
    public NerMention e1;
    public NerMention e2;
    public String relation;
    public String comments;
    
    public SemEval2010Sentence() {
        
    }
    
    public void intern() {
        id = id.intern();
        QLists.intern(words);
        e1.intern();
        e2.intern();
        relation = relation.intern();
        comments = comments.intern();
    }
    
    public static List<SemEval2010Sentence> fromAnnoSentence(AnnoSentence sent, int i) {
        List<SemEval2010Sentence> seSents = new ArrayList<>();
        for (int k=0; k<sent.getNePairs().size(); k++) {
            SemEval2010Sentence seSent = new SemEval2010Sentence();
            SemEval2010Sentence source;
            if (sent.getSourceSent() != null && sent.getSourceSent() instanceof SemEval2010Sentence) {
                source = (SemEval2010Sentence) sent.getSourceSent();
                seSent.comments = source.comments;
                seSent.id = source.id;
            } else {
                seSent.id = String.valueOf(i);
            }
            seSent.words = sent.getWords();
            seSent.e1 = sent.getNePairs().get(k).get1();
            seSent.e2 = sent.getNePairs().get(k).get2();
            seSent.relation = sent.getRelLabels().get(k);
            seSents.add(seSent);
        }
        return seSents;
    }

    public AnnoSentence toAnnoSentence() {
        AnnoSentence sent = new AnnoSentence();
        sent.setWords(this.words);
        sent.setNamedEntities(new NerMentions(this.words.size(), QLists.getList(e1, e2)));
        RelationMentions rms = new RelationMentions();        
        rms.add(new RelationMention(this.relation, null, QLists.getList(
                new Pair<String, NerMention>("e1", this.e1),
                new Pair<String, NerMention>("e2", this.e2)), 
                null));
        sent.setRelations(rms);
        sent.setSourceSent(this);
        return sent;
    }
    
}
