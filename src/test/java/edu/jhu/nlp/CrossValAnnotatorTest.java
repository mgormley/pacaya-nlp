package edu.jhu.nlp;

import static org.junit.Assert.assertEquals;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Set;

import org.junit.Test;

import edu.jhu.nlp.CrossValAnnotator.TrainableFactory;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.data.simple.AnnoSentenceCollection;
import edu.jhu.nlp.features.TemplateLanguage.AT;
import edu.jhu.pacaya.util.collections.QLists;
import edu.jhu.pacaya.util.collections.QSets;
import edu.jhu.prim.bimap.CountingIntObjectBimap;

public class CrossValAnnotatorTest {

    @Test
    public void testTrainAndAnnotate() throws Exception {
        TrainableFactory factory = () -> new MajorityVote();
        CrossValAnnotator cva = new CrossValAnnotator(factory , 3);
        
        AnnoSentenceCollection trainGold = getTaggedSents("B", "R", "R", "G", "G", "B");
        AnnoSentenceCollection devGold = getTaggedSents("Y", "Y");
        AnnoSentenceCollection trainInput = trainGold.getWithAtsRemoved(QLists.getList(AT.POS));
        AnnoSentenceCollection devInput = devGold.getWithAtsRemoved(QLists.getList(AT.POS));
        cva.trainAndAnnotate(trainInput, trainGold, devInput, devGold);
        
        assertEquals("G", trainInput.get(0).getPosTag(0));
        assertEquals("G", trainInput.get(1).getPosTag(0));
        assertEquals("B", trainInput.get(2).getPosTag(0));
        assertEquals("B", trainInput.get(3).getPosTag(0));
        assertEquals("R", trainInput.get(4).getPosTag(0));
        assertEquals("R", trainInput.get(5).getPosTag(0));
        
        // The tie is broken by the first tag seen by the majority vote.
        assertEquals("B", devInput.get(0).getPosTag(0));
        assertEquals("B", devInput.get(1).getPosTag(0));
    }
    
    public static AnnoSentenceCollection getTaggedSents(String... tags) {
        AnnoSentenceCollection sents = new AnnoSentenceCollection();
        for (int i=0; i<tags.length; i++) {
            sents.add(getTaggedSent(tags[i]));
        }
        return sents;
    }
    
    public static AnnoSentence getTaggedSent(String... tags) {
        AnnoSentence sent = new AnnoSentence();
        sent.setWords(Arrays.asList(tags));
        sent.setPosTags(Arrays.asList(tags));
        return sent;
    }

    private static class MajorityVote implements Trainable {

        private static final long serialVersionUID = 1L;
        private String majorityTag; 
        
        @Override
        public void train(AnnoSentenceCollection trainInput, AnnoSentenceCollection trainGold,
                AnnoSentenceCollection devInput, AnnoSentenceCollection devGold) {
            CountingIntObjectBimap<String> counts = new CountingIntObjectBimap<>();
            for (AnnoSentence sent : trainGold) {
                for (String w : sent.getPosTags()) {
                    counts.lookupIndex(w);
                }
            }
            counts.stopGrowth();
            
            int majorityCount = -1;
            for (int i=0; i<counts.size(); i++) {
                int c = counts.lookupObjectCount(i);
                if (c > majorityCount) {
                    majorityCount = c;
                    majorityTag = counts.lookupObject(i);
                }
            }
        }
        
        @Override
        public void annotate(AnnoSentenceCollection sents) {
            for (AnnoSentence sent : sents) {
                sent.setPosTags(new ArrayList<>());
                for (int i=0; i<sent.size(); i++) {
                    sent.getPosTags().add(majorityTag);
                }
            }
        }

        @Override
        public Set<AT> getAnnoTypes() {
            return QSets.getSet(AT.POS);
        }
        
    }
    
}
