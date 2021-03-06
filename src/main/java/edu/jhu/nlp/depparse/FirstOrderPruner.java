package edu.jhu.nlp.depparse;

import java.io.File;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.AbstractParallelAnnotator;
import edu.jhu.nlp.Annotator;
import edu.jhu.nlp.CorpusStatistics;
import edu.jhu.nlp.data.DepEdgeMask;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.data.simple.AnnoSentenceCollection;
import edu.jhu.nlp.features.TemplateLanguage.AT;
import edu.jhu.nlp.joint.JointNlpDecoder.JointNlpDecoderPrm;
import edu.jhu.nlp.joint.JointNlpFgExamplesBuilder;
import edu.jhu.nlp.joint.JointNlpFgExamplesBuilder.JointNlpFgExampleBuilderPrm;
import edu.jhu.nlp.joint.JointNlpFgModel;
import edu.jhu.pacaya.gm.data.FgExampleList;
import edu.jhu.pacaya.gm.data.LFgExample;
import edu.jhu.pacaya.gm.feat.ObsFeatureConjoiner;
import edu.jhu.pacaya.gm.inf.BeliefPropagation.BeliefPropagationPrm;
import edu.jhu.pacaya.gm.inf.BeliefPropagation.BpScheduleType;
import edu.jhu.pacaya.gm.inf.BeliefPropagation.BpUpdateOrder;
import edu.jhu.pacaya.gm.inf.FgInferencer;
import edu.jhu.pacaya.gm.model.FactorGraph;
import edu.jhu.pacaya.util.Prm;
import edu.jhu.pacaya.util.Threads;
import edu.jhu.pacaya.util.collections.QSets;
import edu.jhu.pacaya.util.files.QFiles;
import edu.jhu.pacaya.util.semiring.LogSemiring;
import edu.jhu.prim.util.Lambda.FnIntToVoid;
import edu.jhu.prim.util.Timer;

public class FirstOrderPruner implements Annotator {

    private static final long serialVersionUID = 1L;
    private static final Logger log = LoggerFactory.getLogger(FirstOrderPruner.class);
    private File pruneModel;
    private JointNlpFgExampleBuilderPrm exPrm;
    private JointNlpDecoderPrm dPrm;

    public FirstOrderPruner(File pruneModel, JointNlpFgExampleBuilderPrm exPrm, JointNlpDecoderPrm dPrm) {
        this.pruneModel = pruneModel;
        this.exPrm = Prm.clone(exPrm);
        this.dPrm = dPrm;
    }
    
    @Override
    public void annotate(final AnnoSentenceCollection inputSents) {
        // Read a model from a file.
        log.info("Reading pruning model from file: " + pruneModel);
        final JointNlpFgModel model = (JointNlpFgModel) QFiles.deserialize(pruneModel);
        
        ObsFeatureConjoiner ofc = model.getOfc();
        CorpusStatistics cs = model.getCs();
        exPrm.fgPrm = model.getFgPrm();   
        
        final BeliefPropagationPrm bpPrm = new BeliefPropagationPrm();
        bpPrm.s = LogSemiring.getInstance();
        bpPrm.schedule = BpScheduleType.TREE_LIKE;
        bpPrm.updateOrder = BpUpdateOrder.SEQUENTIAL;
        bpPrm.normalizeMessages = false;
        bpPrm.maxIterations = 1;
        bpPrm.convergenceThreshold = 0;
        bpPrm.keepTape = false;
        
        // Get unlabeled data.
        JointNlpFgExamplesBuilder builder = new JointNlpFgExamplesBuilder(exPrm, ofc, cs, false);
        final FgExampleList data = builder.getData(inputSents, null);
        
        // Decode and create edge pruning mask.
        log.info("Running the pruning decoder.");
        final AtomicInteger numEdgesTot = new AtomicInteger(0);
        final AtomicInteger numEdgesKept = new AtomicInteger(0);
        Timer timer = new Timer();
        timer.start();
        Threads.forEach(0, inputSents.size(), new FnIntToVoid() {            
            @Override
            public void call(int i) {
                try {
                    LFgExample ex = data.get(i);
                    AnnoSentence inputSent = inputSents.get(i);
                    
                    // Decode.
                    DepEdgeMaskDecoder decoder = new DepEdgeMaskDecoder(dPrm.maskPrm);
                    FactorGraph fgLatPred = ex.getFactorGraph();
                    fgLatPred.updateFromModel(model);
                    FgInferencer infLatPred = bpPrm.getInferencer(fgLatPred);
                    infLatPred.run();
                    DepEdgeMask mask = decoder.decode(infLatPred, ex, inputSent);
                    
                    // Update the pruning mask.
                    if (mask != null) {
                        if (inputSent.getDepEdgeMask() == null) {
                            inputSent.setDepEdgeMask(mask);
                        } else {
                            inputSent.getDepEdgeMask().and(mask);
                        }
                    }
                    numEdgesKept.addAndGet(mask.getCount());
                    int n = inputSent.getWords().size();
                    numEdgesTot.addAndGet(n*n);
                } catch (Throwable t) {
                    AbstractParallelAnnotator.logThrowable(log, t);
                }
            }
        });
        timer.stop();
        log.info(String.format("Pruning decoded at %.2f tokens/sec", inputSents.getNumTokens() / timer.totSec()));
        int numEdgesPruned = numEdgesTot.get() - numEdgesKept.get();
        log.info(String.format("Pruned %d / %d = %f edges", numEdgesPruned, numEdgesTot.get(), 
                (double) numEdgesPruned / numEdgesTot.get()));  
    }

    @Override
    public Set<AT> getAnnoTypes() {
        return QSets.getSet(AT.DEP_EDGE_MASK);
    }
    
}
