package edu.jhu.nlp.data.simple;

import java.io.IOException;

import edu.jhu.nlp.AnnoPipeline;
import edu.jhu.nlp.Annotator;
import edu.jhu.nlp.EvalPipeline;
import edu.jhu.nlp.Evaluator;
import edu.jhu.nlp.Trainable;
import edu.jhu.nlp.eval.ProportionAnnotated;

/**
 * This class defines the standard use case of a CorpusHandler combined with an AnnoPipeline.
 * 
 * @author mgormley
 */
public class Experiments {

    /**
     * Runs a standard sequence of training, annotation, and evaluation on data from a CorpusHandler (corpus). 
     * 
     * @param corpus The input data.
     * @param anno The annotation pipeline to be trained.
     * @param eval The evaluation pipeline.
     * @throws IOException
     */
    public static void trainAnnoEval(CorpusHandler corpus, Trainable anno, Evaluator eval)
            throws IOException {
        trainAnnoEvalPrepGold(corpus, anno, eval, new AnnoPipeline());
    }
    
    /**
     * Runs a standard sequence of preprocessing, training, annotation, and evaluation on data from a CorpusHandler (corpus). 
     * 
     * @param corpus The input data.
     * @param anno The annotation pipeline to be trained.
     * @param eval The evaluation pipeline.
     * @param prepGold The preprocessing pipeline which is only applied to gold data. (null indicates no preprocessing should be done).
     * @throws IOException
     */
    public static void trainAnnoEvalPrepGold(CorpusHandler corpus, Trainable anno, Evaluator eval, Annotator prepGold)
            throws IOException {
        {
            // Either of train or dev might be null.
            AnnoSentenceCollection trainGold = corpus.getTrainGold();
            AnnoSentenceCollection trainInput = corpus.getTrainInput();
            AnnoSentenceCollection devGold = corpus.getDevGold();
            AnnoSentenceCollection devInput = corpus.getDevInput();
    
            if (corpus.hasTrain()) {
                // Preprocess the gold train data.
                prepGold.annotate(trainGold);
                // Write out the gold train data.
                corpus.writeTrainGold();
            }
            if (corpus.hasDev()) {
                // Preprocess the gold dev data.
                prepGold.annotate(devGold);
                // Write out the gold dev data.
                corpus.writeDevGold();
            }
            
            if (corpus.hasTrain()) {
                // Train a model.
                anno.trainAndAnnotate(trainInput, trainGold, devInput, devGold);
            } else if (corpus.hasDev()) { // but not train
                anno.annotate(devInput);
            }
            
            if (corpus.hasTrain()) {
                // Decode and evaluate the train data.
                corpus.writeTrainPreds(trainInput);
                eval.evaluate(trainInput, trainGold, "train");
                corpus.clearTrainCache();
            }
            if (corpus.hasDev()) {
                // Decode and evaluate the dev data.
                corpus.writeDevPreds(devInput);
                eval.evaluate(devInput, devGold, "dev");
                corpus.clearDevCache();
            }
        }
        
        if (corpus.hasTest()) {
            // Decode test data.
            String name = "test";
            AnnoSentenceCollection testInput = corpus.getTestInput();
            anno.annotate(testInput);
            corpus.writeTestPreds(testInput);
            if (corpus.hasTestGold()) {
                AnnoSentenceCollection testGold = corpus.getTestGold();
                // Preprocess the gold test data.
                prepGold.annotate(testGold);
                // Write out the gold test data.
                corpus.writeTestGold();
                // Evaluate test data.
                eval.evaluate(testInput, testGold, name);
            } else {
                (new ProportionAnnotated(CorpusHandler.getPredAts())).evaluate(testInput, null, name);
            }
            corpus.clearTestCache();
        }
    }

}
