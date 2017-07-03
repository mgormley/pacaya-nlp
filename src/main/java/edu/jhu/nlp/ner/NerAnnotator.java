package edu.jhu.nlp.ner;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.hlt.optimize.function.Function;
import edu.jhu.nlp.AbstractParallelAnnotator;
import edu.jhu.nlp.Annotator;
import edu.jhu.nlp.Evaluator;
import edu.jhu.nlp.Trainable;
import edu.jhu.nlp.data.simple.AlphabetStore;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.data.simple.AnnoSentenceCollection;
import edu.jhu.nlp.data.simple.IntAnnoSentence;
import edu.jhu.nlp.eval.NerEvaluator;
import edu.jhu.nlp.features.TemplateLanguage.AT;
import edu.jhu.nlp.ner.NerFactorGraphBuilder.NerFactorGraph;
import edu.jhu.nlp.ner.NerFactorGraphBuilder.NerFactorGraphBuilderPrm;
import edu.jhu.pacaya.gm.data.FgExampleList;
import edu.jhu.pacaya.gm.data.LFgExample;
import edu.jhu.pacaya.gm.data.LabeledFgExample;
import edu.jhu.pacaya.gm.data.UFgExample;
import edu.jhu.pacaya.gm.data.UnlabeledFgExample;
import edu.jhu.pacaya.gm.decode.MbrDecoder;
import edu.jhu.pacaya.gm.decode.MbrDecoder.MbrDecoderPrm;
import edu.jhu.pacaya.gm.model.FgModel;
import edu.jhu.pacaya.gm.model.VarConfig;
import edu.jhu.pacaya.gm.train.CrfTrainer;
import edu.jhu.pacaya.gm.train.CrfTrainer.CrfTrainerPrm;
import edu.jhu.pacaya.util.Prm;
import edu.jhu.pacaya.util.Threads;
import edu.jhu.pacaya.util.collections.QSets;
import edu.jhu.prim.util.Lambda.FnIntToVoid;
import edu.jhu.prim.util.Timer;
import edu.jhu.prim.vector.IntDoubleVector;

public class NerAnnotator implements Trainable {

    private static final long serialVersionUID = 1L;

    public static class NerAnnotatorPrm extends Prm {
        private static final long serialVersionUID = 1L;
        /** The CRF traininer parameters. */
        public CrfTrainerPrm crfPrm = null;
        /** The decoder parameters. */
        public MbrDecoderPrm mbrPrm = null;
        /** NER factor graph settings. */
        public NerFactorGraphBuilderPrm nerPrm = null;
    }
    
    private static final Logger log = LoggerFactory.getLogger(NerAnnotator.class);

    private FgModel model;
    private NerFactorGraphBuilder builder;
    private NerAnnotatorPrm prm;
    private List<String> tagLabelSet;
    private AlphabetStore store;
    
    public NerAnnotator(NerAnnotatorPrm prm) {
        this.prm = prm;
    }
    
    @Override
    public void train(AnnoSentenceCollection trainInput, AnnoSentenceCollection trainGold,
            AnnoSentenceCollection devInput, AnnoSentenceCollection devGold) {
        log.info("Initializing corpus statistics.");
        tagLabelSet = new ArrayList<>(getTagLabelSet(trainGold));
        log.info("Tag set: " + tagLabelSet);
        store = new AlphabetStore(trainInput);
        
        log.info("Initializing model to all zeros.");        
        model = new FgModel(prm.nerPrm.featureHashMod);
        log.info("Num model params: {}", model.getNumParams());
        FgExampleList data = getData(trainInput, trainGold);
        log.info("Training model.");
        CrfTrainer trainer = new CrfTrainer(prm.crfPrm);
        trainer.train(model, data, getValidationFn(devInput, devGold));
    }

    private Set<String> getTagLabelSet(AnnoSentenceCollection trainGold) {
        HashSet<String> labels = new HashSet<>();
        for (AnnoSentence sent : trainGold) {
            labels.addAll(sent.getNeTags());
        }
        return labels;
    }

    @Override
    public void annotate(final AnnoSentenceCollection sents) {
        if (model == null) {
            throw new IllegalStateException("No model exists. Must call train() before annotate().");
        }
        log.debug("Running the decoder");
        Timer timer = new Timer();
        timer.start();
        
        final FgExampleList data = getData(sents, null);  
        // Add the new predictions to the input sentences.
        Threads.forEach(0, sents.size(), new FnIntToVoid() {            
            @Override
            public void call(int i) {
                try {
                    UFgExample ex = data.get(i);
                    AnnoSentence inputSent = sents.get(i);
                    AnnoSentence predSent = decode(model, ex, inputSent);
                    sents.set(i, predSent);
                } catch (Throwable t) {
                    AbstractParallelAnnotator.logThrowable(log, t);
                }
            }
        });
        
        timer.stop();
        log.debug(String.format("Decoded at %.2f tokens/sec with %d threads", sents.getNumTokens() / timer.totSec(), Threads.numThreads));
    }

    @Override
    public Set<AT> getAnnoTypes() {
        return QSets.getSet(AT.NE_TAGS);
    }

    private FgExampleList getData(final AnnoSentenceCollection inputSents, final AnnoSentenceCollection goldSents) {
        return new FgExampleList() {
            
            @Override
            public int size() {
                return inputSents.size();
            }
            
            @Override
            public LFgExample get(int i) {
                NerFactorGraphBuilder builder = new NerFactorGraphBuilder(prm.nerPrm);
                // Construct a factor graph which carries the builder.
                NerFactorGraph fg = new NerFactorGraph(builder);
                IntAnnoSentence isent = new IntAnnoSentence(inputSents.get(i), store);
                builder.build(isent, fg, tagLabelSet);
                VarConfig goldConfig = null;
                if (goldSents != null) {
                    goldConfig = new VarConfig();
                    builder.addVarAssignments(goldSents.get(i).getNeTags(), goldConfig);
                    return new LabeledFgExample(fg, goldConfig);
                } else {
                    return new UnlabeledFgExample(fg);
                }                
            }
        };
    }

    private AnnoSentence decode(FgModel model, UFgExample ex, AnnoSentence inputSent) {
        NerFactorGraphBuilder builder = ((NerFactorGraph) ex.getFactorGraph()).getBuilder();
        MbrDecoder mbrDecoder = new MbrDecoder(prm.mbrPrm);
        mbrDecoder.decode(model, ex);
        List<String> neTags = builder.getTagsFromMbrVarConfig(mbrDecoder.getMbrVarConfig());
        AnnoSentence predSent = inputSent.getShallowCopy();
        predSent.setNeTags(neTags);
        return predSent;
    }

    // TODO: This is ugly. Can we improve this interface?
    private Function getValidationFn(AnnoSentenceCollection devInput, AnnoSentenceCollection devGold) {
        if (devInput == null || devGold == null) { return null; }
        final Annotator anno = this;
        final Evaluator eval = new NerEvaluator();
        return new Function() {
            
            @Override
            public double getValue(IntDoubleVector point) {
                AnnoSentenceCollection devPred = devInput.getShallowCopy();
                anno.annotate(devPred);
                return eval.evaluate(devPred, devGold, "dev");
            }
            
            @Override
            public int getNumDimensions() {
                return -1;
            }
        };
    }

}
