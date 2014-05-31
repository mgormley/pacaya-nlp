package edu.jhu.autodiff.erma;

import java.util.List;

import edu.jhu.autodiff.AbstractTensorModule;
import edu.jhu.autodiff.Exp;
import edu.jhu.autodiff.Module;
import edu.jhu.autodiff.ScalarDivide;
import edu.jhu.autodiff.Tensor;
import edu.jhu.autodiff.TensorIdentity;
import edu.jhu.autodiff.TopoOrder;
import edu.jhu.util.collections.Lists;
import edu.jhu.util.semiring.Algebra;

/** 
 * Performs softmax MBR decoding for dependency parsing.
 * 
 * 1. Compute edge weights w_e = exp(p_{\theta}(y_e=1|x) / T)
 * 2. Run inside-outside on w_e to get q_{\theta}^{1/T}(e)
 * 
 * The input to this module is expected to be a tensor containing the edge weights for dependency
 * parsing. The tensor is expected to be an nxn matrix, capable of being converted to EdgeScores
 * internally by EdgeScores.tensorToEdgeScores().
 * 
 * @author mgormley
 */
public class SoftmaxMbrDepParse implements Module<Tensor> {
    
    private Module<Tensor> pIn;
    private double temperature;
    private Algebra s;
    private TopoOrder topo;
    
    public SoftmaxMbrDepParse(Module<Tensor> margIn, double temperature, Algebra s) {
        this.pIn = margIn;
        this.temperature = temperature;
        this.s = s;
    }
    
    @Override
    public Tensor forward() {
        topo = new TopoOrder();
        
        TensorIdentity ti = new TensorIdentity(Tensor.getScalarTensor(temperature));
        topo.add(ti);
        ScalarDivide divide = new ScalarDivide(pIn, ti, 0);
        topo.add(divide);
        Exp exp = new Exp(divide);
        topo.add(exp);
        // TODO: convert between non-semiring numbers and semiring numbers.
        InsideOutsideDepParse io = new InsideOutsideDepParse(exp, s);
        topo.add(io);
        
        return topo.forward();
    }

    @Override
    public void backward() {
        topo.backward();
    }

    @Override
    public List<Module<Tensor>> getInputs() {
        return Lists.getList(pIn);
    }

    @Override
    public Tensor getOutput() {
        return topo.getOutput();
    }

    @Override
    public Tensor getOutputAdj() {
        return topo.getOutputAdj();
    }

    @Override
    public void zeroOutputAdj() {
        topo.zeroOutputAdj();
    }
    
}