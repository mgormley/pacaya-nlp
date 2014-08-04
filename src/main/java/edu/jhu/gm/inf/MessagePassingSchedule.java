package edu.jhu.gm.inf;

import java.util.List;

import edu.jhu.gm.model.FactorGraph.FgEdge;

public interface MessagePassingSchedule {

    /**
     * Gets the order for a message passing algorithm (e.g. belief propagation). An order consists
     * of a list of edges, global factors, or lists of these two. When a global factor is included
     * in the order, this indicates that all the messages from that global factor to its variables
     * should be sent. 
     * <br><br>
     * Note that some algorithms might not support orders which include an edge from
     * a global factor to a variable.
     * 
     * @return The order of messages to send.
     */
    List<FgEdge> getOrder();

}
