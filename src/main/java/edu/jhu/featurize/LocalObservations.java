package edu.jhu.featurize;

import edu.jhu.parse.cky.Rule;

/**
 * The local observations. The union of this class with the sentence / corpus stats member variables
 * of the {@link TemplateFeatureExtractor} form the full set of observed variables.
 * 
 * @author mgormley
 */
public class LocalObservations {
    
    public static final int UNDEF_INT = Integer.MIN_VALUE;
    private int pidx = UNDEF_INT;
    private int cidx = UNDEF_INT;
    private int midx = UNDEF_INT;
    private Rule rule = null;
    private int rStartIdx = UNDEF_INT;
    private int rMidIdx = UNDEF_INT;
    private int rEndIdx = UNDEF_INT;

    private LocalObservations() {
    }

    public LocalObservations(int pidx, int cidx, int midx, Rule rule, int rStartIdx, int rMidIdx, int rEndIdx) {
        super();
        this.pidx = pidx;
        this.cidx = cidx;
        this.midx = midx;
        this.rule = rule;
        this.rStartIdx = rStartIdx;
        this.rMidIdx = rMidIdx;
        this.rEndIdx = rEndIdx;
    }

    public int getPidx() {
        return getIfDefined(pidx, "parent index");
    }

    public int getCidx() {
        return getIfDefined(cidx, "child index");
    }

    public int getMidx() {
        return getIfDefined(midx, "modifier index");
    }

    public Rule getRule() {
        return getIfDefined(rule, "rule");
    }

    public int getRStartIdx() {
        return getIfDefined(rStartIdx, "Start of a rule span");
    }

    public int getRMidIdx() {
        return getIfDefined(rMidIdx, "Split point for a rule span");
    }

    public int getREndIdx() {
        return getIfDefined(rEndIdx, "End of a rule span");
    }

    private int getIfDefined(int idx, String obsName) {
        if (idx == UNDEF_INT) {
            throw new IllegalStateException("Local observation undefined: " + obsName);
        }
        return idx;
    }

    private <T> T getIfDefined(T obj, String obsName) {
        if (obj == null) {
            throw new IllegalStateException("Local observation undefined: " + obsName);
        }
        return obj;
    }

    /* ---------- Factory Methods ----------- */
    public static LocalObservations newPidxCidx(int pidx, int cidx) {
        LocalObservations pi = new LocalObservations();
        pi.pidx = pidx;
        pi.cidx = cidx;
        return pi;
    }

    public static LocalObservations newPidxCidxMidx(int pidx, int cidx, int midx) {
        LocalObservations pi = new LocalObservations();
        pi.pidx = pidx;
        pi.cidx = cidx;
        pi.midx = midx;
        return pi;
    }

    public static LocalObservations newPidx(int pidx) {
        LocalObservations pi = new LocalObservations();
        pi.pidx = pidx;
        return pi;
    }

    public static LocalObservations newRule(Rule rule) {
        LocalObservations pi = new LocalObservations();
        pi.rule = rule;
        return pi;
    }

    public static LocalObservations newRuleStartMidEnd(Rule r, int start, int mid, int end) {
        LocalObservations pi = new LocalObservations();
        pi.rule = r;
        pi.rStartIdx = start;
        pi.rMidIdx = mid;
        pi.rEndIdx = end;
        return pi;
    }
}