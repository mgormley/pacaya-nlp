package edu.jhu.parse.dmv;

import edu.jhu.data.Sentence;
import edu.jhu.data.conll.ValidParentsSentence;
import edu.jhu.parse.cky.Rule;
import edu.jhu.parse.cky.chart.ConstrainedChartCell.ChartCellConstraint;
import edu.jhu.parse.dmv.DmvRule.DmvRuleType;

public class DmvChartCellConstraint implements ChartCellConstraint {

    /** Sentence which also indicates whether a parent/child arc is valid. */
    private ValidParentsSentence sent;
    
    public DmvChartCellConstraint() {
        this.sent = null;
    }

    /** @inheritDoc */
    @Override
    public void setSentence(Sentence sentence) {
        if (!(sentence instanceof ValidParentsSentence)) {
            throw new IllegalStateException("Sentence must have valid parent annotations.");
        }
        this.sent = (ValidParentsSentence)sentence;
    }
    
    /** @inheritDoc */
    @Override
    public boolean shouldUpdateCell(int start, int end, int mid, Rule r, double score) {

        final DmvRule dmvRule = (DmvRule)r;
        if (dmvRule.getType() == DmvRuleType.STRUCTURAL) {
            // This is only true for structural rules.
            int leftHead = start / 2;
            int rightHead = end / 2;
            // Determine which is the head and child.
            boolean isLeftHead = dmvRule.isLeftHead();
            int head = isLeftHead ? leftHead : rightHead;
            int child = isLeftHead ? rightHead : leftHead;
            
            // Check that the constraints allow this arc.
            if (!sent.isValid(child, head)) {
                // Do not update this chart cell.
                return false;
            }
        } else if (dmvRule.getType() == DmvRuleType.ROOT) {
            int head = -1;
            int child = mid / 2;
            // Check that the constraints allow this arc.
            if (!sent.isValid(child, head)) {
                // Do not update this chart cell.
                return false;
            }
        }
        // Update this chart cell.
        return true;
    }
    
}