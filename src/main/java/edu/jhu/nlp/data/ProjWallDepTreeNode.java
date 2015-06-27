package edu.jhu.nlp.data;

import edu.jhu.pacaya.parse.dep.ParentsArray;


public class ProjWallDepTreeNode extends ProjDepTreeNode {
    
    public ProjWallDepTreeNode() {
        super(WallDepTreeNode.WALL_LABEL);
        setPosition(ParentsArray.WALL_POSITION);
    }

    @Override
    public boolean isWall() {
        return true; 
    }
    
}
