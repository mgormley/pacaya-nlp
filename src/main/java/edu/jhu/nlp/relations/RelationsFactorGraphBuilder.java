package edu.jhu.nlp.relations;

import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.CorpusStatistics;
import edu.jhu.nlp.ObsFeTypedFactor;
import edu.jhu.nlp.data.NerMention;
import edu.jhu.nlp.data.Span;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.embed.Embeddings;
import edu.jhu.nlp.fcm.FcmFactor;
import edu.jhu.nlp.relations.RelObsFeatures.EntityTypeRepl;
import edu.jhu.nlp.relations.RelObsFeatures.RelObsFePrm;
import edu.jhu.nlp.relations.RelWordFeatures.EmbFeatType;
import edu.jhu.nlp.relations.RelWordFeatures.RelWordFeaturesPrm;
import edu.jhu.pacaya.gm.feat.ObsFeatureConjoiner;
import edu.jhu.pacaya.gm.model.FactorGraph;
import edu.jhu.pacaya.gm.model.Var;
import edu.jhu.pacaya.gm.model.Var.VarType;
import edu.jhu.pacaya.gm.model.VarSet;
import edu.jhu.pacaya.util.FeatureNames;
import edu.jhu.pacaya.util.Prm;
import edu.jhu.pacaya.util.cli.Opt;
import edu.jhu.prim.tuple.Pair;

public class RelationsFactorGraphBuilder {

    private static final Logger log = LoggerFactory.getLogger(RelationsFactorGraphBuilder.class);

    public static class RelationsFactorGraphBuilderPrm extends Prm {
        // TODO: Cleanup these names: drop "use" and add "rel" prefix.
        private static final long serialVersionUID = 1L;
        @Opt(hasArg=true, description="The type of the relation variables.")
        public VarType relVarType = VarType.PREDICTED;
        @Opt(hasArg=true, description="Whether to use the standard binary features.")
        public boolean useZhou05Features = true;
        @Opt(hasArg=true, description="Whether to use the embedding FCM features.")
        public boolean useEmbeddingFeatures = true;
        @Opt(hasArg=true, description="Whether to use fine tuning for the FCM.")
        public boolean fcmFineTuning = false;
        @Opt(hasArg=true, description="The feature set for embeddings.")
        public EmbFeatType embFeatType = EmbFeatType.FULL;   
        @Opt(hasArg=true, description="What to replace removed entity types with.")
        public EntityTypeRepl entityTypeRepl = EntityTypeRepl.NONE;        
    }
    
    public enum RelationFactorType {
        RELATION
    }
    
    private RelationsFactorGraphBuilderPrm prm;
    private List<RelVar> relVars;
    
    public RelationsFactorGraphBuilder(RelationsFactorGraphBuilderPrm prm) {
        this.prm = prm;
    }
    
    public static class RelVar extends Var {

        private static final long serialVersionUID = 1L;

        NerMention ment1;
        NerMention ment2;     
        
        public RelVar(VarType type, String name, NerMention ment1, NerMention ment2, List<String> stateNames) {
            super(type, stateNames.size(), name, stateNames);
            if (ment1.compareTo(ment2) >= 0) {
                log.warn("The first mention (ment1) should always preceed the second mention (ment2)");
            }
            this.ment1 = ment1;
            this.ment2 = ment2;
        }

        public static String getDefaultName(Span arg1, Span arg2) {
            return String.format("RelVar_[%d,%d]_[%d,%d]", arg1.start(), arg2.end(), arg2.start(), arg2.end());
        }
        
    }
    
    /**
     * Adds factors and variables to the given factor graph.
     */
    public void build(AnnoSentence sent, ObsFeatureConjoiner ofc, FactorGraph fg, CorpusStatistics cs) {
        relVars = new ArrayList<>();
        
        // Create relation variables.
        //
        // Iterate over all pairs of mentions, such that ne1 comes before ne2.
        // This code assumes that the mentions are already in sorted order.
        List<RelVar> rvs = new ArrayList<>();
        if (sent.getNePairs() == null) {
            throw new IllegalArgumentException("Relation extraction requires named entity pairs.");
        }
    	for (Pair<NerMention,NerMention> pair : sent.getNePairs()) {
    		NerMention ne1 = pair.get1();
    		NerMention ne2 = pair.get2();
            // Create relation variable.
            String name = RelVar.getDefaultName(ne1.getSpan(), ne2.getSpan());
            RelVar rv = new RelVar(prm.relVarType, name, ne1, ne2, cs.relationStateNames);
            rvs.add(rv);
            relVars.add(rv);
        }
            	
        // Exponential family factor's feature extractor.
        RelObsFePrm relPrm = new RelObsFePrm();
        relPrm.entityTypeRepl = prm.entityTypeRepl;
        relPrm.useZhou05Features = prm.useZhou05Features;
        RelObsFeatures relFe = new RelObsFeatures(relPrm, sent, ofc.getTemplates());
    	
    	// FCM's feature extractor.
    	RelWordFeatures wordFe = null;
    	if (prm.useEmbeddingFeatures) {
    	    RelWordFeaturesPrm wordPrm = new RelWordFeaturesPrm();
            wordPrm.embFeatType = prm.embFeatType;
            wordPrm.entityTypeRepl = prm.entityTypeRepl;
            // HACK: Does this work correctly?
            final FeatureNames alphabet = ofc.fcmAlphabet;
            wordFe = new RelWordFeatures(wordPrm, sent, alphabet);
    	}
    	
        // Create unary factors for each relation variable.
        for (RelVar rv : rvs) {
            VarSet vars = new VarSet(rv);
            // Even if the interesting features are turned off, we still want the bias feature from this factor.
            fg.addFactor(new ObsFeTypedFactor(vars, RelationFactorType.RELATION, ofc, relFe));
            if (prm.useEmbeddingFeatures) {
                // HACK: The embeddings should be carried in a submodel.
                Embeddings embeddings = (Embeddings)ofc.embeddings;
                fg.addFactor(new FcmFactor(vars, sent, embeddings, ofc, prm.fcmFineTuning, wordFe));
            }
        }
    }
    
    public List<RelVar> getRelVars() {
        return relVars;
    }
    
}
