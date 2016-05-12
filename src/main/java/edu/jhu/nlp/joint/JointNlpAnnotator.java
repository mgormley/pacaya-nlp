package edu.jhu.nlp.joint;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashSet;
import java.util.Set;
import java.util.zip.GZIPOutputStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.hlt.optimize.function.Function;
import edu.jhu.nlp.AbstractParallelAnnotator;
import edu.jhu.nlp.CorpusStatistics;
import edu.jhu.nlp.CorpusStatistics.CorpusStatisticsPrm;
import edu.jhu.nlp.Evaluator;
import edu.jhu.nlp.Trainable;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.data.simple.AnnoSentenceCollection;
import edu.jhu.nlp.data.simple.CorpusHandler;
import edu.jhu.nlp.embed.Embeddings;
import edu.jhu.nlp.eval.DepParseAccuracy;
import edu.jhu.nlp.eval.LabelEvaluator;
import edu.jhu.nlp.eval.PosTagAccuracy;
import edu.jhu.nlp.eval.RelationEvaluator;
import edu.jhu.nlp.eval.SprlEvaluator;
import edu.jhu.nlp.eval.SrlEvaluator;
import edu.jhu.nlp.eval.SrlEvaluator.SrlEvaluatorPrm;
import edu.jhu.nlp.fcm.FcmModule;
import edu.jhu.nlp.features.TemplateLanguage.AT;
import edu.jhu.nlp.joint.JointNlpDecoder.JointNlpDecoderPrm;
import edu.jhu.nlp.joint.JointNlpFgExamplesBuilder.JointNlpFgExampleBuilderPrm;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.RoleStructure;
import edu.jhu.pacaya.gm.data.FgExampleList;
import edu.jhu.pacaya.gm.data.LFgExample;
import edu.jhu.pacaya.gm.data.LibDaiFgIo;
import edu.jhu.pacaya.gm.data.UFgExample;
import edu.jhu.pacaya.gm.feat.FactorTemplateList;
import edu.jhu.pacaya.gm.feat.ObsFeatureConjoiner;
import edu.jhu.pacaya.gm.feat.ObsFeatureConjoiner.ObsFeatureConjoinerPrm;
import edu.jhu.pacaya.gm.model.FactorGraph;
import edu.jhu.pacaya.gm.train.CrfTrainer;
import edu.jhu.pacaya.gm.train.CrfTrainer.CrfTrainerPrm;
import edu.jhu.pacaya.util.Prm;
import edu.jhu.pacaya.util.Threads;
import edu.jhu.pacaya.util.collections.QSets;
import edu.jhu.pacaya.util.files.QFiles;
import edu.jhu.prim.util.Lambda.FnIntToVoid;
import edu.jhu.prim.util.Timer;
import edu.jhu.prim.vector.IntDoubleVector;

/**
 * Joint annotator for SRL and syntactic dependency parsing.
 *
 * @author mgormley
 */
public class JointNlpAnnotator implements Trainable {

    public static enum InitParams { UNIFORM, RANDOM };
    private static final long serialVersionUID = 1L;

    public static class JointNlpAnnotatorPrm extends Prm {
        private static final long serialVersionUID = 1L;

        public JointNlpFgExampleBuilderPrm buPrm = new JointNlpFgExampleBuilderPrm();
        public JointNlpDecoderPrm dePrm = new JointNlpDecoderPrm();
        // How to initialize the parameters of the model
        public InitParams initParams = InitParams.UNIFORM;
        // Whether to skip punctuation when evaluating dependency parsing.
        public boolean dpSkipPunctuation = false;
        // --------------------------------------------------------------------
        // These parameters are only used if a NEW model is created. If a model
        // is loaded from disk, these are ignored.
        public transient CrfTrainerPrm crfPrm = new CrfTrainerPrm();
        public CorpusStatisticsPrm csPrm = new CorpusStatisticsPrm();
        public ObsFeatureConjoinerPrm ofcPrm = new ObsFeatureConjoinerPrm();
        // if srl and sprl are both predicted, only validate with srl
        public boolean favorSrlValidation = false;
        // if srl and sprl are both predicted, only validate with sprl
        // only one of these can be true; if both are false and both srl and sprl are
        // predicted, then the harmonic mean is used
        public boolean favorSprlValidation = false;
        // We also ignore buPrm.fePrm, which is overwritten by the value in the model.
        // --------------------------------------------------------------------
        // for debugging (path if any to export the train factor graphs be written to libdai format before and after training)
        public String exportTrainToLibDAI = null;
    }

    private static final Logger log = LoggerFactory.getLogger(JointNlpAnnotator.class);
    private JointNlpAnnotatorPrm prm;
    private JointNlpFgModel model = null;
    private Embeddings embeddings; // TODO: Remove this hack.

    public JointNlpAnnotator(JointNlpAnnotatorPrm prm, Embeddings embeddings) {
        this.prm = prm;
        this.embeddings = embeddings;
    }

    private void maybeWriteGraphs(FgExampleList data, JointNlpFgModel model, String category) {
        if (prm.exportTrainToLibDAI != null) {
            Path outDir = Paths.get(prm.exportTrainToLibDAI, category);
            // create directory if doesn't already exist
            if (!Files.exists(outDir)) {
                try {
                    Files.createDirectories(outDir);
                } catch (IOException e) {
                    e.printStackTrace();
                    return;
                }
            }
            // write out each fg
            log.info(String.format("Writing libdai formatted factor graph files to: %s",  outDir.toAbsolutePath().toString()));
            for (int i = 0; i < data.size(); i++) {
                LFgExample example = data.get(i);
                FactorGraph fg = example.getFactorGraph();
                fg.updateFromModel(model);
                LibDaiFgIo.write(fg, Paths.get(outDir.toString(), String.format("%d.fg", i)));
            }
        }
    }

    @Override
    public void train(AnnoSentenceCollection trainInput, AnnoSentenceCollection trainGold,
            AnnoSentenceCollection devInput, AnnoSentenceCollection devGold) {
        log.info("Initializing data.");
        CorpusStatistics cs;
        ObsFeatureConjoiner ofc;
        if (model == null) {
            FactorTemplateList fts = new FactorTemplateList();
            ofc = new ObsFeatureConjoiner(prm.ofcPrm, fts);
            cs = new CorpusStatistics(prm.csPrm);
        } else {
            ofc = model.getOfc();
            cs = model.getCs();
            ofc.getTemplates().startGrowth();
        }
        ofc.embeddings = embeddings;
        JointNlpFgExamplesBuilder builder = new JointNlpFgExamplesBuilder(prm.buPrm, ofc, cs, true);
        FgExampleList data = builder.getData(trainInput, trainGold);

        if (model == null) {
            model = new JointNlpFgModel(cs, ofc, prm.buPrm.fgPrm);
            if (prm.initParams == InitParams.RANDOM) {
                log.info("Initializing model with zero-one Gaussian.");
                model.setRandomStandardNormal();
            } else if (prm.initParams == InitParams.UNIFORM) {
                log.info("Initializing model to all zeros.");
                // Do nothing.
            } else {
                throw new RuntimeException("Parameter initialization method not implemented: " + prm.initParams);
            }

            if (embeddings != null) {
                log.info("Initializing the model with word embeddings.");
                FcmModule.initModelWithEmbeds(embeddings, model, ofc);
            }
        } else {
            log.info("Using read model as initial parameters for training.");
        }
        log.info(String.format("Num model params: %d", model.getNumParams()));

        log.info("Training model.");
        CrfTrainer trainer = new CrfTrainer(prm.crfPrm);
        maybeWriteGraphs(data, model, "before");
        trainer.train(model, data, getValidationFn(devInput, devGold, "dev"));
        maybeWriteGraphs(data, model, "after");
        ofc.getTemplates().stopGrowth();
    }

    public Function getValidationFn(final AnnoSentenceCollection devInput, final AnnoSentenceCollection devGold, String devName) {
        if (devInput == null || devGold == null) { return null; }
        final JointNlpAnnotator anno = this;
        final Evaluator eval;
        if (CorpusHandler.getPredAts().equals(QSets.getSet(AT.DEP_TREE))) {
            eval = new DepParseAccuracy(prm.dpSkipPunctuation);
        } else if (CorpusHandler.getPredAts().equals(QSets.getSet(AT.POS))) {
            eval = new PosTagAccuracy();
        } else if (CorpusHandler.getPredAts().equals(QSets.getSet(AT.SRL)) ||
                CorpusHandler.getPredAts().equals(QSets.getSet(AT.SRL_PRED_IDX, AT.SRL))
                ) {
            eval = getSrlEvaluator();
        } else if (CorpusHandler.getPredAts().equals(QSets.getSet(AT.REL_LABELS)) ||
                CorpusHandler.getPredAts().equals(QSets.getSet(AT.RELATIONS)) ||
                CorpusHandler.getPredAts().equals(QSets.getSet(AT.REL_LABELS, AT.RELATIONS))
                ) {
            eval = new RelationEvaluator();
        } else if (CorpusHandler.getPredAts().equals(QSets.getSet(AT.SPRL))) {
            eval = getSprlEvaluator();
        } else if (CorpusHandler.getPredAts().contains(AT.SPRL) && CorpusHandler.getPredAts().contains(AT.SRL)) {
            // check the option
            if (prm.favorSrlValidation) {
                assert !prm.favorSprlValidation;
                eval = getSrlEvaluator();
            } else if (prm.favorSprlValidation) {
                assert !prm.favorSrlValidation;
                eval = getSprlEvaluator();
            } else {
                final Evaluator srlEval = getSrlEvaluator();
                final SprlEvaluator sprlEval = getSprlEvaluator();
                eval = new Evaluator() {

                    @Override
                    public double evaluate(AnnoSentenceCollection predSents, AnnoSentenceCollection goldSents,
                            String name) {
                        double srlF1 = Math.abs(srlEval.evaluate(predSents, goldSents, name));
                        double sprlF1 = Math.abs(sprlEval.evaluate(predSents, goldSents, name));
                        double f1 = LabelEvaluator.harmonicMean(srlF1, sprlF1);
                        return -f1;
                    }

                };
            }

        } else {
            log.warn("Validation function not implemented. Skipping.");
            return null;
        }
        JointNlpFgExamplesBuilder builder = new JointNlpFgExamplesBuilder(prm.buPrm, model.getOfc(), model.getCs(), false);
        final FgExampleList devData = builder.getData(devInput, null);
        return new Function() {

            @Override
            public double getValue(IntDoubleVector point) {
                AnnoSentenceCollection devPred = devInput.getShallowCopy();
                anno.annotate(devPred, devData);
                return eval.evaluate(devPred, devGold, devName);
            }

            @Override
            public int getNumDimensions() {
                return -1;
            }
        };
    }

    private SprlEvaluator getSprlEvaluator() {
        // make an evaluator that doesn't break it down by category and that doesn't report 
        return new SprlEvaluator(prm.buPrm.fgPrm.sprlPrm.roleStructure, prm.buPrm.fgPrm.sprlPrm.allowPredArgSelfLoops, false, false, false);
    }

    private Evaluator getSrlEvaluator() {
        final Evaluator eval;
        SrlEvaluatorPrm evalPrm = new SrlEvaluatorPrm();
        evalPrm.evalPredSense = prm.buPrm.fgPrm.srlPrm.predictSense;
        evalPrm.evalPredPosition = prm.buPrm.fgPrm.srlPrm.predictPredPos;
        evalPrm.evalRoles = (prm.buPrm.fgPrm.srlPrm.roleStructure != RoleStructure.NO_ROLES);
        eval = new SrlEvaluator(evalPrm);
        return eval;
    }

    @Override
    public void annotate(AnnoSentenceCollection sents) {
        if (model == null) {
            throw new IllegalStateException("No model exists. Must call train() or loadModel() before annotate().");
        }

        log.info("Running the decoder");
        Timer timer = new Timer();
        timer.start();
        JointNlpFgExamplesBuilder builder = new JointNlpFgExamplesBuilder(prm.buPrm, model.getOfc(), model.getCs(), false);
        FgExampleList data = builder.getData(sents, null);
        annotate(sents, data);
        timer.stop();
        log.info(String.format("Decoded at %.2f tokens/sec with %d threads", sents.getNumTokens() / timer.totSec(), Threads.numThreads));
    }

    private void annotate(final AnnoSentenceCollection sents, final FgExampleList data) {
        // Add the new predictions to the input sentences.
        Threads.forEach(0, sents.size(), new FnIntToVoid() {
            @Override
            public void call(int i) {
                try {
                    UFgExample ex = data.get(i);
                    AnnoSentence inputSent = sents.get(i);
                    JointNlpDecoder decoder = new JointNlpDecoder(prm.dePrm);
                    AnnoSentence predSent = decoder.decode(model, ex, inputSent);
                    sents.set(i, predSent);
                } catch (Throwable t) {
                    AbstractParallelAnnotator.logThrowable(log, t);
                }
            }
        });
    }

    public JointNlpFgModel getModel() {
        return model;
    }

    public void loadModel(File modelIn) {
        // Read a model from a file.
        log.info("Reading model from file: " + modelIn);
        loadModel((JointNlpFgModel) QFiles.deserialize(modelIn));
    }

    public void loadModel(JointNlpFgModel model) {
        this.model = model;
        // Restore the model settings from the serialized model.
        prm.buPrm.fgPrm = model.getFgPrm();
    }

    public void saveModel(File modelOut) {
        // Write the model to a file.
        log.info("Serializing model to file: " + modelOut);
        QFiles.serialize(model, modelOut);
    }

    public void printModel(File printModel) throws IOException {
        // Print the model to a file.
        log.info("Printing human readable model to file: " + printModel);
        OutputStream os = new FileOutputStream(printModel);
        if (printModel.getName().endsWith(".gz")) {
            os = new GZIPOutputStream(os);
        }
        Writer writer = new BufferedWriter(new OutputStreamWriter(os, "UTF-8"));
        model.printModel(writer);
        writer.close();
    }

    @Override
    public Set<AT> getAnnoTypes() {
        HashSet<AT> ats = new HashSet<>();
        if (prm.buPrm.fgPrm.includeDp) {
            ats.add(AT.DEP_TREE);
        }
        if (prm.buPrm.fgPrm.includeSrl) {
            ats.add(AT.SRL);
        }
        if (prm.buPrm.fgPrm.includeRel) {
            ats.add(AT.REL_LABELS);
        }
        return ats;
    }

}
