package edu.jhu.nlp.data;

import edu.jhu.pacaya.parse.dep.ParentsArray;


public class WallDepTreeNode extends NonprojDepTreeNode {

    public static final String WALL_ID = "__WALL__";
    public static final String WALL_LABEL = WALL_ID;
    
    public WallDepTreeNode() {
        super(WALL_LABEL, ParentsArray.WALL_POSITION);
    }

    @Override
    public boolean isWall() {
        return true; 
    }
    
}
