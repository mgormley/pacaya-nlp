package edu.jhu.nlp.joint;

import java.util.Date;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.hlt.optimize.AdaDelta;
import edu.jhu.hlt.optimize.AdaDelta.AdaDeltaPrm;
import edu.jhu.hlt.optimize.AdaGradComidL1;
import edu.jhu.hlt.optimize.AdaGradComidL1.AdaGradComidL1Prm;
import edu.jhu.hlt.optimize.AdaGradComidL2;
import edu.jhu.hlt.optimize.AdaGradComidL2.AdaGradComidL2Prm;
import edu.jhu.hlt.optimize.AdaGradSchedule;
import edu.jhu.hlt.optimize.AdaGradSchedule.AdaGradSchedulePrm;
import edu.jhu.hlt.optimize.BottouSchedule;
import edu.jhu.hlt.optimize.BottouSchedule.BottouSchedulePrm;
import edu.jhu.hlt.optimize.LBFGS;
import edu.jhu.hlt.optimize.LBFGS_port.LBFGSPrm;
import edu.jhu.hlt.optimize.Optimizer;
import edu.jhu.hlt.optimize.SGD;
import edu.jhu.hlt.optimize.SGD.SGDPrm;
import edu.jhu.hlt.optimize.SGDFobos;
import edu.jhu.hlt.optimize.SGDFobos.SGDFobosPrm;
import edu.jhu.hlt.optimize.function.BatchFunctionOpts;
import edu.jhu.hlt.optimize.function.DifferentiableBatchFunction;
import edu.jhu.hlt.optimize.function.DifferentiableFunction;
import edu.jhu.hlt.optimize.function.DifferentiableFunctionOpts;
import edu.jhu.pacaya.util.cli.Opt;
import edu.jhu.prim.tuple.Pair;

public class OptimizerFactory {

    public static enum OptimizerType { LBFGS, SGD, ADAGRAD, ADAGRAD_COMID, ADADELTA, FOBOS, ASGD }

    private static final Logger log = LoggerFactory.getLogger(OptimizerFactory.class);

    // Options for optimization.
    @Opt(hasArg=true, description="The optimization method to use for training.")
    public static OptimizerType optimizer = OptimizerType.ADAGRAD_COMID;
    @Opt(hasArg=true, description="The weight on the L1 regularizer.")
    public static double l1Lambda = 0.0;
    @Opt(hasArg=true, description="The weight on the L2 regularizer.")
    public static double l2Lambda = 1.0;
    @Opt(hasArg=true, description="Max iterations for L-BFGS training.")
    public static int lbfgsMaxIters = 1000;
    @Opt(hasArg=true, description="Number of iterations to cache for LBFGS training.")
    public static int lbfgsCachedIters = 6;
    @Opt(hasArg=true, description="Number of effective passes over the dataset for SGD.")
    public static int sgdNumPasses = 30;
    @Opt(hasArg=true, description="The batch size to use at each step of SGD.")
    public static int sgdBatchSize = 15;
    @Opt(hasArg=true, description="The initial learning rate for SGD.")
    public static double sgdInitialLr = 0.1;
    @Opt(hasArg=true, description="Whether to sample with replacement for SGD.")
    public static boolean sgdWithRepl = false;
    @Opt(hasArg=true, description="Whether to automatically select the learning rate.")
    public static boolean sgdAutoSelectLr = true;
    @Opt(hasArg=true, description="How many epochs between auto-select runs.")
    public static int sgdAutoSelecFreq = 5;
    @Opt(hasArg=true, description="Whether to compute the function value on iterations other than the last.")
    public static boolean sgdComputeValueOnNonFinalIter = true;
    @Opt(hasArg=true, description="Whether to do parameter averaging.")
    public static boolean sgdAveraging = false;
    @Opt(hasArg=true, description="Whether to do early stopping.")
    public static boolean sgdEarlyStopping = true;
    @Opt(hasArg=true, description="The AdaGrad parameter for scaling the learning rate.")
    public static double adaGradEta = 0.1;
    @Opt(hasArg=true, description="The constant addend for AdaGrad.")
    public static double adaGradConstantAddend = 1e-9;
    @Opt(hasArg=true, description="The initial value of the sum of squares for AdaGrad.")
    public static double adaGradInitialSumSquares = 0;
    @Opt(hasArg=true, description="The decay rate for AdaDelta.")
    public static double adaDeltaDecayRate = 0.95;
    @Opt(hasArg=true, description="The constant addend for AdaDelta.")
    public static double adaDeltaConstantAddend = Math.pow(Math.E, -6.);
    @Opt(hasArg=true, description="Stop training by this date/time.")
    public static Date stopTrainingBy = null;

    public static Pair<Optimizer<DifferentiableFunction>, Optimizer<DifferentiableBatchFunction>> getOptimizers() {
        if (OptimizerFactory.stopTrainingBy != null && new Date().after(OptimizerFactory.stopTrainingBy)) {
            log.warn("Training will never begin since stopTrainingBy has already happened: " + OptimizerFactory.stopTrainingBy);
            log.warn("Ignoring stopTrainingBy by setting it to null.");
            OptimizerFactory.stopTrainingBy = null;
        }
        
        Optimizer<DifferentiableFunction> opt;
        Optimizer<DifferentiableBatchFunction> batchOpt;
        if (optimizer == OptimizerType.LBFGS) {
            LBFGSPrm prm = new LBFGSPrm();
            prm.max_iterations = lbfgsMaxIters;
            prm.m = lbfgsCachedIters;
            opt = DifferentiableFunctionOpts.getRegularizedOptimizer(new LBFGS(prm), l1Lambda, l2Lambda);            
            batchOpt = null;
        } else if (optimizer == OptimizerType.SGD || optimizer == OptimizerType.ASGD  ||
                optimizer == OptimizerType.ADAGRAD || optimizer == OptimizerType.ADADELTA) {
            opt = null;
            SGDPrm sgdPrm = getSgdPrm();
            if (optimizer == OptimizerType.SGD){
                BottouSchedulePrm boPrm = new BottouSchedulePrm();
                boPrm.initialLr = sgdInitialLr;
                boPrm.lambda = l1Lambda + l2Lambda;
                sgdPrm.sched = new BottouSchedule(boPrm);
            } else if (optimizer == OptimizerType.ASGD){
                BottouSchedulePrm boPrm = new BottouSchedulePrm();
                boPrm.initialLr = sgdInitialLr;
                boPrm.lambda = l1Lambda + l2Lambda;
                boPrm.power = 0.75;
                sgdPrm.sched = new BottouSchedule(boPrm);
                sgdPrm.averaging = true;
            } else if (optimizer == OptimizerType.ADAGRAD){
                AdaGradSchedulePrm adaGradPrm = new AdaGradSchedulePrm();
                adaGradPrm.eta = adaGradEta;
                adaGradPrm.constantAddend = adaGradConstantAddend;
                adaGradPrm.initialSumSquares = adaGradInitialSumSquares;
                sgdPrm.sched = new AdaGradSchedule(adaGradPrm);
            } else if (optimizer == OptimizerType.ADADELTA){
                AdaDeltaPrm adaDeltaPrm = new AdaDeltaPrm();
                adaDeltaPrm.decayRate = adaDeltaDecayRate;
                adaDeltaPrm.constantAddend = adaDeltaConstantAddend;
                sgdPrm.sched = new AdaDelta(adaDeltaPrm);
                sgdPrm.autoSelectLr = false;
            }
            batchOpt = BatchFunctionOpts.getRegularizedOptimizer(new SGD(sgdPrm), l1Lambda, l2Lambda);
        } else if (optimizer == OptimizerType.ADAGRAD_COMID) {
            if (l1Lambda > 0 && l2Lambda > 0) {
                // TODO: Implement this optimizer.
                throw new RuntimeException("ADAGRAD_COMID with l1lambda > 0 && l2lambda > 0 is not yet implemented");
            }
            if (l1Lambda > 0) {
                AdaGradComidL1Prm sgdPrm = new AdaGradComidL1Prm();
                setSgdPrm(sgdPrm);
                sgdPrm.l1Lambda = l1Lambda;
                sgdPrm.eta = adaGradEta;
                sgdPrm.constantAddend = adaGradConstantAddend;
                sgdPrm.initialSumSquares = adaGradInitialSumSquares;
                sgdPrm.sched = null;
                opt = null;
                batchOpt = new AdaGradComidL1(sgdPrm);
            } else {
                AdaGradComidL2Prm sgdPrm = new AdaGradComidL2Prm();
                setSgdPrm(sgdPrm);
                sgdPrm.l2Lambda = l2Lambda;
                sgdPrm.eta = adaGradEta;
                sgdPrm.constantAddend = adaGradConstantAddend;
                sgdPrm.initialSumSquares = adaGradInitialSumSquares;
                sgdPrm.sched = null;
                opt = null;
                batchOpt = new AdaGradComidL2(sgdPrm);
            }
        } else if (optimizer == OptimizerType.FOBOS) {
            SGDFobosPrm sgdPrm = new SGDFobosPrm();
            setSgdPrm(sgdPrm);
            sgdPrm.l1Lambda = l1Lambda;
            sgdPrm.l2Lambda = l2Lambda;
            BottouSchedulePrm boPrm = new BottouSchedulePrm();
            boPrm.initialLr = sgdInitialLr;
            boPrm.lambda = l1Lambda + l2Lambda;
            sgdPrm.sched = new BottouSchedule(boPrm);  
            opt = null;
            batchOpt = new SGDFobos(sgdPrm);
        } else {
            throw new RuntimeException("Optimizer not supported: " + optimizer);
        }        
        return new Pair<>(opt, batchOpt);
    }
    
    private static SGDPrm getSgdPrm() {
        SGDPrm prm = new SGDPrm();
        setSgdPrm(prm);
        return prm;
    }

    private static void setSgdPrm(SGDPrm prm) {
        prm.numPasses = sgdNumPasses;
        prm.batchSize = sgdBatchSize;
        prm.withReplacement = sgdWithRepl;
        prm.stopBy = stopTrainingBy;
        prm.autoSelectLr = sgdAutoSelectLr;
        prm.autoSelectFreq = sgdAutoSelecFreq;
        prm.computeValueOnNonFinalIter = sgdComputeValueOnNonFinalIter;
        prm.averaging = sgdAveraging; 
        prm.earlyStopping = sgdEarlyStopping; 
        // Make sure we correctly set the schedule somewhere else.
        prm.sched = null;
    }

    
}
