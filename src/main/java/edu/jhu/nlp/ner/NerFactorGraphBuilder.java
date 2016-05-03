package edu.jhu.nlp.ner;

import java.util.ArrayList;
import java.util.List;

import edu.jhu.nlp.data.simple.IntAnnoSentence;
import edu.jhu.nlp.tag.BitshiftTokenFeatures;
import edu.jhu.nlp.tag.HashObsFeatsFactor;
import edu.jhu.pacaya.gm.feat.FeatureVector;
import edu.jhu.pacaya.gm.model.FactorGraph;
import edu.jhu.pacaya.gm.model.Var;
import edu.jhu.pacaya.gm.model.Var.VarType;
import edu.jhu.pacaya.gm.model.VarConfig;
import edu.jhu.pacaya.gm.model.VarSet;
import edu.jhu.pacaya.util.Prm;
import edu.jhu.prim.util.SafeCast;

public class NerFactorGraphBuilder {

    public static class NerFactorGraphBuilderPrm extends Prm {
        private static final long serialVersionUID = 1L;
        /** The value of the mod for use in the feature hashing trick. If <= 0, feature-hashing will be disabled. */
        public int featureHashMod = 1048576; // 2^20
        /** Whether to include unary factors on each tag. */
        public boolean unigramFactors = true;
        /** Whether to include binary factors on each pair of tags. */
        public boolean bigramFactors = true;
    }
    
    /** Carrier for the builder. */
    public static class NerFactorGraph extends FactorGraph {
        private static final long serialVersionUID = 1L;
        private NerFactorGraphBuilder builder;
        public NerFactorGraph(NerFactorGraphBuilder builder) { this.builder = builder; }
        public NerFactorGraphBuilder getBuilder() { return builder; }
    }

    // TODO: These should move to a shared constants class, so as to avoid conflicts 
    // with other HashObsFeatsFactors (e.g. for POS tagging).
    private static final short UNIGRAM_FACTOR = 4;
    private static final short BIGRAM_FACTOR = 5;
    
    private NerFactorGraphBuilderPrm prm;
    private List<Var> tagVars;
    
    public NerFactorGraphBuilder(NerFactorGraphBuilderPrm prm) {
        this.prm = prm;
    }

    /**
     * Adds factors and variables to the given factor graph.
     */
    public void build(IntAnnoSentence isent, FactorGraph fg, List<String> tagLabelSet) {
        // Create tag variables.
        tagVars = new ArrayList<>();
        for (int i=0; i<isent.size(); i++) {
            Var v = new Var(VarType.PREDICTED, tagLabelSet.size(), "tag"+i, tagLabelSet);
            tagVars.add(v);
        }
        // Create factors.
        addFactors(isent, fg);
    }

    private void addFactors(IntAnnoSentence isent, FactorGraph fg) {
        for (int i=0; i<isent.size(); i++) {
            if (prm.unigramFactors) {
                // Unary factor for each tag.
                VarSet vars = new VarSet(tagVars.get(i));
                final FeatureVector obsFeats = new FeatureVector();
                BitshiftTokenFeatures.addUnigramFeatures(isent, i, obsFeats, -1, (short) 0); // config=0
                fg.addFactor(new HashObsFeatsFactor(vars, obsFeats, UNIGRAM_FACTOR, prm.featureHashMod));
            }
            if (i > 0 && prm.bigramFactors) {
                // Binary factor for each pair of tags.
                VarSet vars = new VarSet(tagVars.get(i-1), tagVars.get(i));
                final FeatureVector obsFeats = new FeatureVector();
                BitshiftTokenFeatures.addBigramFeatures(isent, i, obsFeats, -1, (short) 0); // config=0
                fg.addFactor(new HashObsFeatsFactor(vars, obsFeats, BIGRAM_FACTOR, prm.featureHashMod));
            }
        }
    }
    
    public List<Var> getTagVars() {
        return tagVars;
    }
    
    /* ------------------------- Encode ------------------------- */
    public void addVarAssignments(List<String> tags, VarConfig vc) {
        for (int i=0; i<tagVars.size(); i++) {
            Var var = tagVars.get(i);
            vc.put(var, tags.get(i));
        }
    }
    
    /* ------------------------- Decode ------------------------- */
    public List<String> getTagsFromMbrVarConfig(VarConfig mbrVarConfig) {
        ArrayList<String> tags = new ArrayList<>();
        for (int i=0; i<tagVars.size(); i++) {
            Var var = tagVars.get(i);
            tags.add(mbrVarConfig.getStateName(var));
        }
        return tags;
    }
    
}
