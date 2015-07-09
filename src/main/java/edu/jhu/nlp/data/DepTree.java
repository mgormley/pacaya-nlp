package edu.jhu.nlp.data;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.pacaya.nlp.data.Sentence;
import edu.jhu.pacaya.parse.dep.ParentsArray;
import edu.jhu.prim.bimap.IntObjectBimap;

public class DepTree implements Iterable<DepTreeNode> {

    private static final Logger log = LoggerFactory.getLogger(DepTree.class);
    protected List<DepTreeNode> nodes = new ArrayList<DepTreeNode>();
    protected int[] parents;
    protected boolean isProjective;
    protected final boolean isSingleHeaded = true;
    
    protected DepTree() {
        // Only for subclasses.
    }
    
    /**
     * Construct a dependency tree from a sentence and the head of each token.
     * 
     * @param sentence The input sentence.
     * @param parents The index of the parent of each token. -1 indicates the root.
     * @param isProjective Whether the tree is projective.
     */
    public DepTree(Sentence sentence, int[] parents, boolean isProjective) {
        this.isProjective = isProjective;
        this.parents = parents;
        nodes.add(new WallDepTreeNode());
        for (int i=0; i<sentence.size(); i++) {
            String label = sentence.get(i);
            nodes.add(new NonprojDepTreeNode(label, i));
        }
        // Add parent/child links to DepTreeNodes
        addParentChildLinksToNodes();
    }
    
    /**
     * Construct a dependency tree from a wall node and its children.
     * 
     * @param wall
     */
    @SuppressWarnings("unchecked")
    public DepTree(ProjDepTreeNode wall) {
        isProjective = true;
        nodes = (List<DepTreeNode>)wall.getInorderTraversal();
        // Set all the positions on the nodes
        int position;
        position=ParentsArray.WALL_POSITION;
        for (DepTreeNode node : nodes) {
            ((ProjDepTreeNode)node).setPosition(position);
            position++;
        }
        // Set all the parent positions
        parents = new int[nodes.size()-1];
        for (int i=0; i<parents.length; i++) {
            ProjDepTreeNode parent = (ProjDepTreeNode)nodes.get(i+1).getParent();
            if (parent == null) {
                parents[i] = ParentsArray.EMPTY_POSITION;
            } else {
                parents[i] = parent.getPosition();
            }
        }
        checkTree();
    }

    protected DepTreeNode getNodeByPosition(int position) {
        return nodes.get(position+1);
    }
    
    protected void addParentChildLinksToNodes() {
        checkTree();
        for (int i=0; i<parents.length; i++) {
            NonprojDepTreeNode child = (NonprojDepTreeNode)getNodeByPosition(i);
            NonprojDepTreeNode parent = (NonprojDepTreeNode)getNodeByPosition(parents[i]);
            child.setParent(parent);
            parent.addChild(child);
        }
    }

    protected void checkTree() {
        // Check that there is exactly one node with the WALL as its parent
        int emptyCount = ParentsArray.countChildrenOf(parents, ParentsArray.EMPTY_POSITION);
        if (emptyCount != 0) {
            throw new IllegalStateException("Found an empty parent cell. emptyCount=" + emptyCount);
        }
        int wallCount = ParentsArray.countChildrenOf(parents, ParentsArray.WALL_POSITION);
        if (isSingleHeaded && wallCount != 1) {
            log.warn("There must be exactly one node with the wall as a parent. wallCount=" + wallCount);
        } else if (wallCount < 1) {
            log.warn("There must be some node with the wall as a parent. wallCount=" + wallCount);
        }
        
        // Check that there are no cyles
        if (!ParentsArray.isConnectedAndAcyclic(parents)) {
            throw new IllegalStateException("Found cycle in parents array");
        }

        // Check for proper list lengths
        if (nodes.size()-1 != parents.length) {
            throw new IllegalStateException("Number of nodes does not equal number of parents");
        }
        
        // Check for projectivity if necessary
        if (isProjective) {
            if (!ParentsArray.isProjective(parents)) {
                throw new IllegalStateException("Found non-projective arcs in tree");
            }
        }
    }

    @Override
    public String toString() {
        return nodes.toString();
    }

    public DepTreeNode getWallNode() {
        return nodes.get(0);
    }
    
    public List<DepTreeNode> getNodes() {
        return nodes;
    }

    public Iterator<DepTreeNode> iterator() {
        return nodes.iterator();
    }
    
    /**
     * For testing only.
     * @return
     */
    public int[] getParents() {
        return parents;
    }
    
    public int getNumTokens() {
        return parents.length;
    }

    public Sentence getSentence(IntObjectBimap<String> alphabet) {
        return new DTWrappedSentence(alphabet, this);
    }
    
    // This class adds an alternative constructor to sentence.
    private static class DTWrappedSentence extends Sentence {
        
        private static final long serialVersionUID = 1L;

        public DTWrappedSentence(IntObjectBimap<String> alphabet, DepTree tree) {
            super(alphabet);
            for (DepTreeNode node : tree.getNodes()) {
                if (!node.isWall()) {
                    add(node.getLabel());
                }
            }
        }
        
    }    
    
}
