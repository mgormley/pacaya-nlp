package edu.jhu.nlp.features;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.pacaya.parse.dep.ParentsArray;
import edu.jhu.prim.list.IntArrayList;
import edu.jhu.prim.tuple.Pair;

/**
 * Cache of features for a pair of words in a sentence (parent/predicate and
 * child/argument).
 * 
 * @author mmitchell
 * @author mgormley
 */
public class FeaturizedTokenPair {
    
    /* Feature constructor based on CoNLL 2009:
     * "Multilingual Dependency Learning:
     * A Huge Feature Engineering Method to Semantic Dependency Parsing"
     * Hai Zhao, Wenliang Chen, Chunyu Kit, Guodong Zhou 
     * and
     * "Multilingual Semantic Role Labeling"
     * Anders Bjo ̈rkelund, Love Hafdell, Pierre Nugues
     * Treats features as combinations of feature templates, for:
     * 1. word form (formFeats)
     * 2. lemma (lemmaFeats)
     * 3. part-of-speech (tagFeats)
     * 4. morphological features (morphFeats)
     * 5. syntactic dependency label (deprelFeats)
     * 6. children (childrenFeats)
     * 7. dependency paths (pathFeats)
     * 8. 'high' and 'low' support, siblings, parents (syntacticConnectionFeats).
     */    

    private int pidx;
    private int aidx;
    private FeaturizedToken pTok;
    private FeaturizedToken aTok;    
    private int[] parents;
    
    private IntArrayList linePath;
    private IntArrayList btwnPath;
    private List<Pair<Integer, ParentsArray.Dir>> dependencyPath;
    private List<Pair<Integer, ParentsArray.Dir>> dpPathShare;
    private List<Pair<Integer, ParentsArray.Dir>> dpPathPred;
    private List<Pair<Integer, ParentsArray.Dir>> dpPathArg;
    
    public FeaturizedTokenPair(int pidx, int aidx, FeaturizedToken pTok, FeaturizedToken aTok, AnnoSentence sent) {
        assert pTok.getSent() == aTok.getSent();
        assert pTok.getSent() == sent;
        this.parents = sent.getParents();
        this.pTok = pTok;
        this.aTok = aTok;
        this.pidx = pidx;
        this.aidx = aidx;
        /* ZHAO:  Path. There are two basic types of path between the predicate and the argument candidates. 
         * One is the linear path (linePath) in the sequence, the other is the path in the syntactic 
         * parsing tree (dpPath). For the latter, we further divide it into four sub-types by 
         * considering the syntactic root, dpPath is the full path in the syntactic tree. */
    }
    
    // ------------------------ Getters and Caching Methods ------------------------ //
        
    public List<Pair<Integer,ParentsArray.Dir>> getDependencyPath() {
        if (dependencyPath == null) {
            this.dependencyPath = ParentsArray.getDependencyPath(pidx, aidx, parents);
        }
        return dependencyPath;
    }
    
    public List<Pair<Integer, ParentsArray.Dir>> getDpPathPred() {
        ensureDpPathShare();
        return dpPathPred;
    }
    
    public List<Pair<Integer, ParentsArray.Dir>> getDpPathArg() {
        ensureDpPathShare();
        return this.dpPathArg;
    }
    
    public List<Pair<Integer, ParentsArray.Dir>> getDpPathShare() {
        ensureDpPathShare();
        return dpPathShare;
    }
    
    private void ensureDpPathShare() {
        if (dpPathShare != null) {
            return;
        }        
        this.dpPathShare = new ArrayList<Pair<Integer,ParentsArray.Dir>>();
        /* ZHAO:  Leading two paths to the root from the predicate and the argument, respectively, 
         * the common part of these two paths will be dpPathShare. */
        List<Pair<Integer, ParentsArray.Dir>> argRootPath = aTok.getRootPath();
        List<Pair<Integer, ParentsArray.Dir>> predRootPath = pTok.getRootPath();
        if (argRootPath != null && predRootPath != null) {
            int i = argRootPath.size() - 1;
            int j = predRootPath.size() - 1;
            Pair<Integer,ParentsArray.Dir> argP = argRootPath.get(i);
            Pair<Integer,ParentsArray.Dir> predP = predRootPath.get(j);
            while (argP.equals(predP)) {
                this.dpPathShare.add(argP);
                if (i == 0 || j == 0) {
                    break;
                }
                i--;
                j--;
                argP = argRootPath.get(i);
                predP = predRootPath.get(j);
            }
        }
        /* ZHAO:  Assume that dpPathShare starts from a node r', 
         * then dpPathPred is from the predicate to r', and dpPathArg is from the argument to r'. */
        // Reverse, so path goes towards the root.
        Collections.reverse(this.dpPathShare);
        int r;
        if (this.dpPathShare.isEmpty()) {
            r = -1;
            this.dpPathPred = new ArrayList<Pair<Integer, ParentsArray.Dir>>();
            this.dpPathArg = new ArrayList<Pair<Integer, ParentsArray.Dir>>();
        } else {
            r = this.dpPathShare.get(0).get1();
            this.dpPathPred = ParentsArray.getDependencyPath(pidx, r, parents);
            this.dpPathArg = ParentsArray.getDependencyPath(aidx, r, parents);
            assert this.dpPathPred != null;
            assert this.dpPathArg != null;
        }
    }
    
    public IntArrayList getLinePath() {
        if (linePath == null) {
            cacheLinePath();
        }
        return linePath;
    }
    
    private void cacheLinePath() {
        this.linePath = new IntArrayList();        
        if (pidx < aidx) {
            for (int i=pidx; i<=aidx; i++) {
                this.linePath.add(i);
            }
        } else {
            for (int i=pidx; i>=aidx; i--) {
                this.linePath.add(i);
            }
        }        
    }

    public IntArrayList getBtwnPath() {
        if (btwnPath == null) {
            cacheBtwnPath();
        }
        return btwnPath;
    }
    
    private void cacheBtwnPath() {
        this.btwnPath = new IntArrayList();        
        if (pidx < aidx) {
            for (int i=pidx+1; i<aidx; i++) {
                this.btwnPath.add(i);
            }
        } else {
            for (int i=aidx+1; i<pidx; i++) {
                this.btwnPath.add(i);
            }
        }        
    }
    
    public enum GeneologicalRelation {
        SELF, PARENT, CHILD, SIBLING, ANCESTOR, DESCENDENT, COUSIN
    }
    
    public GeneologicalRelation getGeneologicalRelation() {
        if (pidx == aidx) {
            return GeneologicalRelation.SELF;
        } else if (hasParent(aidx) && pidx == parents[aidx]) {
            return GeneologicalRelation.PARENT;
        } else if (hasParent(pidx) && parents[pidx] == aidx) {
            return GeneologicalRelation.CHILD;
        } else if (hasParent(aidx) && hasParent(pidx) && parents[pidx] == parents[aidx]) {
            return GeneologicalRelation.SIBLING;
        } else if (pidx == -1) {
            return GeneologicalRelation.ANCESTOR; // Short circuit to avoid ArrayIndexOutOfBounds.
        } else if (aidx == -1) {
            return GeneologicalRelation.DESCENDENT;   // Short circuit to avoid ArrayIndexOutOfBounds.
        } else if (ParentsArray.isAncestor(pidx, aidx, parents)) {
            return GeneologicalRelation.ANCESTOR;
        } else if (ParentsArray.isAncestor(aidx, pidx, parents)) {
            return GeneologicalRelation.DESCENDENT;
        } else {
            return GeneologicalRelation.COUSIN;
        }
    }
    
    private boolean hasParent(int aidx) {
        return 0 <= aidx && aidx < parents.length;
    }

    public enum RelativePosition { ON, BEFORE, AFTER };    
    
    public RelativePosition getRelativePosition() {
        if (pidx == aidx) {
            return RelativePosition.ON;
        } else if (pidx < aidx) {
            return RelativePosition.BEFORE;
        } else {
            return RelativePosition.AFTER;
        }
    }

    public int getCountOfNonConsecutivesInPath() {
        List<Pair<Integer,ParentsArray.Dir>> path = getDependencyPath();
        int count = 0;
        if (path != null && path.size() > 0) {
            for (int i=1; i<path.size(); i++) {
                int current = path.get(i).get1();
                int previous = path.get(i-1).get1();
                if (Math.abs(current - previous) != 1) {
                    count++;
                }
            }
        }
        return count;
    }
    
}
