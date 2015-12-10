package edu.jhu.nlp.depparse;

import java.util.Arrays;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.pacaya.gm.app.Decoder;
import edu.jhu.pacaya.gm.data.UFgExample;
import edu.jhu.pacaya.gm.inf.FgInferencer;
import edu.jhu.pacaya.gm.model.FactorGraph;
import edu.jhu.pacaya.gm.model.Var;
import edu.jhu.pacaya.gm.model.VarTensor;
import edu.jhu.pacaya.gm.model.globalfac.LinkVar;
import edu.jhu.pacaya.hypergraph.depparse.InsideOutsideDepParse;
import edu.jhu.pacaya.parse.dep.EdgeScores;
import edu.jhu.pacaya.parse.dep.ParentsArray;
import edu.jhu.pacaya.parse.dep.ProjectiveDependencyParser;

/**
 * Decodes from the marginals of a factor graph for dependency parsing to an int[] representing
 * the parent for each token.
 * 
 * This computes the MBR tree under an accuracy loss function. Note that this is similar to Smith &
 * Eisner (2008)'s proposed method of decoding, but corrects a bug in their paper.
 * 
 * @author mgormley
 */
public class DepParseDecoder implements Decoder<AnnoSentence, int[]> {

    private static final Logger log = LoggerFactory.getLogger(DepParseDecoder.class);

    /**
     * Decodes by computing the MBR tree under an accuracy loss function.
     */
    @Override
    public int[] decode(FgInferencer inf, UFgExample ex, AnnoSentence sent) {
        FactorGraph fg = ex.getFactorGraph();
        int n = sent.size();        
        // Build up the beliefs about the link variables (if present),
        // and compute the MBR dependency parse.
        EdgeScores scores = DepParseDecoder.getEdgeScores(inf, fg, n);
        return DepParseDecoder.getParents(scores);
    }

    /**
     * Get MBR parse, by finding the argmax tree where we treat the score of a tree as the sum of
     * the edge scores.
     */
    public static int[] getParents(EdgeScores scores) {
        // 
        int n = scores.root.length;
        int[] parents = new int[n];
        Arrays.fill(parents, ParentsArray.EMPTY_POSITION);
        if (InsideOutsideDepParse.singleRoot) {
            ProjectiveDependencyParser.parseSingleRoot(scores.root, scores.child, parents);
        } else {
            ProjectiveDependencyParser.parseMultiRoot(scores.root, scores.child, parents);
        }
        return parents;
    }

    // Package-private for DepEdgeMaskDecoder.
    static EdgeScores getEdgeScores(FgInferencer inf, FactorGraph fg, int n) {
        List<Var> vars = fg.getVars();
        int linkVarCount = 0;
        EdgeScores scores = new EdgeScores(n, Double.NEGATIVE_INFINITY);
        for (int varId = 0; varId < vars.size(); varId++) {
            Var var = vars.get(varId);
            VarTensor marg = inf.getMarginals(var);
            if (var instanceof LinkVar) {
                LinkVar link = ((LinkVar)var);
                int c = link.getChild();
                int p = link.getParent();
    
                // Using logOdds is the method of MBR decoding prescribed in Smith &
                // Eisner (2008), but that's a bug in their paper. This breaks the parser
                // when the log-odds are positive infinity.
                // INCORRECT: belief = FastMath.log(marg.getValue(LinkVar.TRUE) / marg.getValue(LinkVar.FALSE));
                double belief = marg.getValue(LinkVar.TRUE);
    
                if (p == -1) {
                    scores.root[c] = belief;
                } else {
                    scores.child[p][c] = belief;
                }
                linkVarCount++;
            }
        }
        if (n*n != linkVarCount) {
            throw new RuntimeException("Currently, EdgeScores only supports decoding all the LinkVars, not a subset.");
        }
        return scores;
    }

}
