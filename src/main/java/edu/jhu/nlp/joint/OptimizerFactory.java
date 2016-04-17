package edu.jhu.nlp.joint;

import java.util.Date;

import org.apache.commons.cli.ParseException;

import edu.jhu.hlt.optimize.AdaDelta;
import edu.jhu.hlt.optimize.AdaDelta.AdaDeltaPrm;
import edu.jhu.hlt.optimize.AdaGradComidL2;
import edu.jhu.hlt.optimize.AdaGradComidL2.AdaGradComidL2Prm;
import edu.jhu.hlt.optimize.AdaGradSchedule;
import edu.jhu.hlt.optimize.AdaGradSchedule.AdaGradSchedulePrm;
import edu.jhu.hlt.optimize.BottouSchedule;
import edu.jhu.hlt.optimize.BottouSchedule.BottouSchedulePrm;
import edu.jhu.hlt.optimize.LBFGS;
import edu.jhu.hlt.optimize.MalletLBFGS;
import edu.jhu.hlt.optimize.MalletLBFGS.MalletLBFGSPrm;
import edu.jhu.hlt.optimize.Optimizer;
import edu.jhu.hlt.optimize.SGD;
import edu.jhu.hlt.optimize.SGD.SGDPrm;
import edu.jhu.hlt.optimize.SGDFobos;
import edu.jhu.hlt.optimize.SGDFobos.SGDFobosPrm;
import edu.jhu.hlt.optimize.StanfordQNMinimizer;
import edu.jhu.hlt.optimize.function.DifferentiableBatchFunction;
import edu.jhu.hlt.optimize.function.DifferentiableFunction;
import edu.jhu.hlt.optimize.function.Regularizer;
import edu.jhu.hlt.optimize.functions.L2;
import edu.jhu.pacaya.util.cli.Opt;
import edu.jhu.prim.tuple.Pair;

public class OptimizerFactory {

    public static enum OptimizerType { LBFGS, LBFGS_MALLET, QN_STANFORD, SGD, ADAGRAD, ADAGRAD_COMID, ADADELTA, FOBOS, ASGD }

    public enum RegularizerType { L2, NONE };
    
    // Options for optimization.
    @Opt(hasArg=true, description="The optimization method to use for training.")
    public static OptimizerType optimizer = OptimizerType.ADAGRAD_COMID;
    @Opt(hasArg=true, description="The variance for the L2 regularizer.")
    public static double l2variance = 1.0;
    @Opt(hasArg=true, description="The type of regularizer.")
    public static RegularizerType regularizer = RegularizerType.NONE;
    @Opt(hasArg=true, description="Max iterations for L-BFGS training.")
    public static int maxLbfgsIterations = 1000;
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
        Optimizer<DifferentiableFunction> opt;
        Optimizer<DifferentiableBatchFunction> batchOpt;
        if (optimizer == OptimizerType.LBFGS) {
            opt = new LBFGS();
            batchOpt = null;
        } else if (optimizer == OptimizerType.LBFGS_MALLET) {
            opt = getMalletLbfgs();
            batchOpt = null;
        } else if (optimizer == OptimizerType.QN_STANFORD) {
            opt = getStanfordLbfgs();
            batchOpt = null;            
        } else if (optimizer == OptimizerType.SGD || optimizer == OptimizerType.ASGD  ||
                optimizer == OptimizerType.ADAGRAD || optimizer == OptimizerType.ADADELTA) {
            opt = null;
            SGDPrm sgdPrm = getSgdPrm();
            if (optimizer == OptimizerType.SGD){
                BottouSchedulePrm boPrm = new BottouSchedulePrm();
                boPrm.initialLr = sgdInitialLr;
                boPrm.lambda = 1.0 / l2variance;
                sgdPrm.sched = new BottouSchedule(boPrm);
            } else if (optimizer == OptimizerType.ASGD){
                BottouSchedulePrm boPrm = new BottouSchedulePrm();
                boPrm.initialLr = sgdInitialLr;
                boPrm.lambda = 1.0 / l2variance;
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
            batchOpt = new SGD(sgdPrm);
        } else if (optimizer == OptimizerType.ADAGRAD_COMID) {
            AdaGradComidL2Prm sgdPrm = new AdaGradComidL2Prm();
            setSgdPrm(sgdPrm);
            //TODO: sgdPrm.l1Lambda = l1Lambda;
            sgdPrm.l2Lambda = 1.0 / l2variance;
            sgdPrm.eta = adaGradEta;
            sgdPrm.constantAddend = adaGradConstantAddend;
            sgdPrm.initialSumSquares = adaGradInitialSumSquares;
            sgdPrm.sched = null;
            opt = null;
            batchOpt = new AdaGradComidL2(sgdPrm);
        } else if (optimizer == OptimizerType.FOBOS) {
            SGDFobosPrm sgdPrm = new SGDFobosPrm();
            setSgdPrm(sgdPrm);
            //TODO: sgdPrm.l1Lambda = l1Lambda;            
            sgdPrm.l2Lambda = 1.0 / l2variance;
            BottouSchedulePrm boPrm = new BottouSchedulePrm();
            boPrm.initialLr = sgdInitialLr;
            boPrm.lambda = 1.0 / l2variance;
            sgdPrm.sched = new BottouSchedule(boPrm);  
            opt = null;
            batchOpt = new SGDFobos(sgdPrm);
        } else {
            throw new RuntimeException("Optimizer not supported: " + optimizer);
        }        
        return new Pair<>(opt, batchOpt);
    }

    public static Regularizer getRegularizer() throws ParseException {
        if (regularizer == RegularizerType.L2) {
            return new L2(l2variance);
        } else if (regularizer == RegularizerType.NONE) {
            return null;
        } else {
            throw new ParseException("Unsupported regularizer: " + regularizer);
        }
    }

    private static edu.jhu.hlt.optimize.Optimizer<DifferentiableFunction> getMalletLbfgs() {
        MalletLBFGSPrm prm = new MalletLBFGSPrm();
        prm.maxIterations = maxLbfgsIterations;
        return new MalletLBFGS(prm);
    }

    private static edu.jhu.hlt.optimize.Optimizer<DifferentiableFunction> getStanfordLbfgs() {
        return new StanfordQNMinimizer(maxLbfgsIterations);
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
