package edu.jhu.nlp.data.simple;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import edu.jhu.nlp.data.DepTree;
import edu.jhu.nlp.data.DepTreebank;
import edu.jhu.nlp.features.TemplateLanguage.AT;
import edu.jhu.pacaya.nlp.data.Sentence;
import edu.jhu.pacaya.nlp.data.SentenceCollection;
import edu.jhu.pacaya.parse.dep.ParentsArray;
import edu.jhu.prim.bimap.IntObjectBimap;

public class AnnoSentenceCollection extends ArrayList<AnnoSentence> {

    private static final long serialVersionUID = -6867088670574667680L;

    /** Stores the source sentences (e.g. the Communication object for Concrete). */
    private Object sourceSents;

    public AnnoSentenceCollection() {
        super();
    }

    public AnnoSentenceCollection(Collection<AnnoSentence> list) {
        super(list);
        if (list instanceof AnnoSentenceCollection) {
            this.sourceSents = ((AnnoSentenceCollection) list).sourceSents;
        }
    }

    public static AnnoSentenceCollection getSingleton(AnnoSentence sent) {
        AnnoSentenceCollection col = new  AnnoSentenceCollection();
        col.add(sent);
        return col;
    }

    public SentenceCollection getWordsAsSentenceCollection(IntObjectBimap<String> alphabet) {
        SentenceCollection sents = new SentenceCollection(alphabet);
        for (AnnoSentence sent : this) {
            List<String> labels = new ArrayList<String>();
            for (String w : sent.getWords()) {
                labels.add(w);
            }
            sents.add(new Sentence(alphabet, labels));
        }
        return sents;
    }

    public SentenceCollection getLemmasAsSentenceCollection(IntObjectBimap<String> alphabet) {
        SentenceCollection sents = new SentenceCollection(alphabet);
        for (AnnoSentence sent : this) {
            List<String> labels = new ArrayList<String>();
            for (String l : sent.getLemmas()) {
                labels.add(l);
            }
            sents.add(new Sentence(alphabet, labels));
        }
        return sents;
    }

    public SentenceCollection getPosTagsAsSentenceCollection(IntObjectBimap<String> alphabet) {
        SentenceCollection sents = new SentenceCollection(alphabet);
        for (AnnoSentence sent : this) {
            List<String> labels = new ArrayList<String>();
            for (String t : sent.getPosTags()) {
                labels.add(t);
            }
            sents.add(new Sentence(alphabet, labels));
        }
        return sents;
    }

    public DepTreebank getPosTagsAndParentsAsDepTreebank(IntObjectBimap<String> alphabet) {
        DepTreebank trees = new DepTreebank(alphabet);
        for (AnnoSentence sent : this) {
            List<String> labels = new ArrayList<String>();
            for (String t : sent.getPosTags()) {
                labels.add(t);
            }
            Sentence sentence = new Sentence(alphabet, labels);
            boolean isProjective = ParentsArray.isProjective(sent.getParents());
            trees.add(new DepTree(sentence, sent.getParents(), isProjective));
        }
        return trees;
    }

    public int getNumTokens() {
        int numTokens = 0;
        for (AnnoSentence sent : this) {
            numTokens += sent.size();
        }
        return numTokens;
    }

    public AnnoSentenceCollection subList(int start, int end) {
        return new AnnoSentenceCollection(super.subList(start, end));
    }

    /**
     * Gets a shallow copy of these sentences with some annotation layers removed.
     * @param removeAts The annotation layers to remove.
     * @return The filtered deep copy.
     */
    public AnnoSentenceCollection getWithAtsRemoved(Collection<AT> removeAts) {
        AnnoSentenceCollection newSents = new AnnoSentenceCollection();
        newSents.sourceSents = this.sourceSents;
        for (AnnoSentence sent : this) {
            AnnoSentence newSent = sent.getShallowCopy();
            newSent.removeAts(removeAts);
            newSents.add(newSent);
        }
        return newSents;
    }

    /** Returns true if some of the sentences have a particular annotation type. */
    public boolean someHaveAt(AT at) {
        for (AnnoSentence sent : this) {
            if (sent.hasAt(at)) {
                return true;
            }
        }
        return false;
    }

    /** Returns true if all of the sentences have a particular annotation type. */
    public boolean allHaveAt(AT at) {
        for (AnnoSentence sent : this) {
            if (!sent.hasAt(at)) {
                return false;
            }
        }
        return true;
    }

    /** Gets the length of the longest sentence in this collection. */
    public int getMaxLength() {
        int maxLen = Integer.MIN_VALUE;
        for (AnnoSentence sent : this) {
            if (sent.size() > maxLen) {
                maxLen = sent.size();
            }
        }
        return maxLen;
    }

    /** Gets the average sentence length. */
    public double getAvgLength() {
        double avgLen = 0;
        for (AnnoSentence sent : this) {
            avgLen += sent.size();
        }
        return avgLen / this.size();
    }

    public static void copyShallow(AnnoSentenceCollection srcSents, AnnoSentenceCollection destSents, AT at) {
        for (int i=0; i<srcSents.size(); i++) {
            AnnoSentence src = srcSents.get(i);
            AnnoSentence dest = destSents.get(i);
            AnnoSentence.copyShallow(src, dest, at);
        }
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i=0; i<size(); i++) {
            sb.append(get(i));
            if (i != size()-1) {
                sb.append(",\n");
            }
        }
        sb.append("]");
        return sb.toString();
    }

    public Object getSourceSents() {
        return sourceSents;
    }

    public void setSourceSents(Object sourceSents) {
        this.sourceSents = sourceSents;
    }

    /**
     * Divides the sentences into numFolds different folds. The remainder is appended to the last fold.
     */
    public List<AnnoSentenceCollection> getFolds(int numFolds) {
        List<AnnoSentenceCollection> folds = new ArrayList<>(numFolds);
        int numSents = this.size();
        int numPerFold = numSents / numFolds;
        int k=-1;
        for (int i=0; i<numSents; i++) {
            if (i%numPerFold == 0 && k < numFolds-1) {
                folds.add(new AnnoSentenceCollection());
                k++;
            }
            folds.get(k).add(this.get(i));
        }
        assert folds.size() == numFolds;
        return folds;
   }

    /**
     * Creates and returns a new AnnoSentenceCollection containing a shallow copy of
     * each sentence in this collection
     */
    public AnnoSentenceCollection getShallowCopy() {
        AnnoSentenceCollection copied = new AnnoSentenceCollection();
        for (AnnoSentence sent : this) {
            copied.add(sent.getShallowCopy());
        }
        return copied;
    }

}
