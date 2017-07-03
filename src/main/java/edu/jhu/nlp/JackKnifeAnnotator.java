package edu.jhu.nlp;

import java.util.List;
import java.util.Set;

import edu.jhu.nlp.data.simple.AnnoSentenceCollection;
import edu.jhu.nlp.features.TemplateLanguage.AT;

/**
 * Annotator wrapper which annotates the training data by dividing into k-folds and annotating each
 * fold with a model trained on the other k-1 folds.
 * 
 * @author mgormley
 */
public class JackKnifeAnnotator implements Trainable {

    private static final long serialVersionUID = 1L;
    private TrainableFactory factory;
    // The primary annotator, trained on the complete training set.
    private Trainable annoPrimary;
    private int numFolds;
    
    public interface TrainableFactory {
        Trainable getInstance();
    }
    
    public JackKnifeAnnotator(TrainableFactory factory, int numFolds) {
        this.factory = factory;
        this.numFolds = numFolds;
        this.annoPrimary = factory.getInstance();
    }

    @Override
    public void train(AnnoSentenceCollection trainInput, AnnoSentenceCollection trainGold,
            AnnoSentenceCollection devInput, AnnoSentenceCollection devGold) {
        throw new UnsupportedOperationException("train() is not supported. Use trainAndAnnotate() instead.");
    }
    
    @Override
    public void trainAndAnnotate(AnnoSentenceCollection trainInput, AnnoSentenceCollection trainGold,
            AnnoSentenceCollection devInput, AnnoSentenceCollection devGold) {
        List<AnnoSentenceCollection> foldsInput = trainInput.getFolds(numFolds);
        List<AnnoSentenceCollection> foldsGold = trainGold.getFolds(numFolds);
        for (int k=0; k<numFolds; k++) {
            AnnoSentenceCollection curInput = new AnnoSentenceCollection();
            AnnoSentenceCollection restInput = new AnnoSentenceCollection();
            AnnoSentenceCollection restGold = new AnnoSentenceCollection();
            for (int j=0; j<numFolds; j++) {
                if (j == k) {
                    curInput.addAll(foldsInput.get(j));
                } else {
                    restInput.addAll(foldsInput.get(j));
                    restGold.addAll(foldsGold.get(j));
                }
            }
            // Train on k-1 folds.
            Trainable anno = factory.getInstance();
            anno.train(restInput, restGold, devInput, devGold);
            // Annotate the current fold.
            anno.annotate(curInput);
        }
        // Train on all folds.
        annoPrimary.train(trainInput, trainGold, devInput, devGold);
        if (devInput != null) {
            annoPrimary.annotate(devInput);
        }
    }

    @Override
    public void annotate(AnnoSentenceCollection sents) {
        annoPrimary.annotate(sents);
    }

    @Override
    public Set<AT> getAnnoTypes() {
        return annoPrimary.getAnnoTypes();
    }

}
