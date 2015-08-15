package edu.jhu.nlp.relations;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.commons.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.data.LabeledSpan;
import edu.jhu.nlp.data.NerMention;
import edu.jhu.nlp.data.Span;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.fcm.FcmModule;
import edu.jhu.nlp.features.FeaturizedSentence;
import edu.jhu.nlp.features.FeaturizedTokenPair;
import edu.jhu.nlp.features.LocalObservations;
import edu.jhu.nlp.features.TemplateFeatureExtractor;
import edu.jhu.nlp.features.TemplateLanguage.EdgeProperty;
import edu.jhu.nlp.features.TemplateLanguage.TokProperty;
import edu.jhu.nlp.relations.RelObsFe.EntityTypeRepl;
import edu.jhu.nlp.relations.RelationsFactorGraphBuilder.RelVar;
import edu.jhu.nlp.tag.BrownClusterTagger;
import edu.jhu.pacaya.gm.data.UFgExample;
import edu.jhu.pacaya.gm.feat.FactorTemplateList;
import edu.jhu.pacaya.gm.feat.FeatureVector;
import edu.jhu.pacaya.gm.feat.ObsFeExpFamFactor;
import edu.jhu.pacaya.gm.feat.ObsFeatureExtractor;
import edu.jhu.pacaya.parse.cky.data.NaryTree;
import edu.jhu.pacaya.parse.dep.ParentsArray;
import edu.jhu.pacaya.util.FeatureNames;
import edu.jhu.pacaya.util.Prm;
import edu.jhu.pacaya.util.cli.Opt;
import edu.jhu.prim.list.IntArrayList;
import edu.jhu.prim.set.IntHashSet;
import edu.jhu.prim.tuple.Pair;
import edu.jhu.prim.util.Lambda.FnObjDoubleToVoid;

/**
 * Per-word features used by the FCM.
 * 
 * @author mgormley
 */
public class WordFeatures {

    public enum EmbFeatType { HEAD_ONLY, HEAD_TYPE, HEAD_TYPE_LOC, HEAD_TYPE_LOC_ST, FULL }

    public static class WordFeaturesPrm extends Prm {
        private static final long serialVersionUID = 1L;
        @Opt(hasArg=true, description="The feature set for embeddings.")
        public EmbFeatType embFeatType = EmbFeatType.FULL;        
        @Opt(hasArg=true, description="What to replace removed entity types with.")
        public EntityTypeRepl entityTypeRepl = EntityTypeRepl.NONE;
    }
    
    private static final Logger log = LoggerFactory.getLogger(WordFeatures.class);

    private WordFeaturesPrm prm;
    private AnnoSentence sent;
    private FeatureNames alphabet;

    public WordFeatures(WordFeaturesPrm prm, AnnoSentence sent, FeatureNames alphabet) {
        this.prm = prm;
        this.sent = sent;
        this.alphabet = alphabet;
    }
    
    public FeatureNames getAlphabet() {
        return alphabet;
    }
    
    public List<FeatureVector> getFeatures(RelVar rv) {
        LocalObservations local = LocalObservations.newNe1Ne2(rv.ment1, rv.ment2);
        // TODO: Do we want a bias feature here?
        RelObsFe.maybeSetEntityTypesAndSubTypes(sent, local, prm.entityTypeRepl);
        List<FeatureVector> fvs = new ArrayList<>();
        for (int i=0; i<sent.size(); i++) {
            fvs.add(new FeatureVector());
        }
        addEmbeddingFeatures(local, fvs);
        return fvs;
    }

    /**
     * Adds the embedding or embeddingless features for the FCM. The "embedding" features include
     * the dimension and value of the embedding. The "embeddingless" features only include the 
     * observed feature with value 1.0, for use in {@link FcmModule}.
     * 
     * @param local The local observations.
     * @param fv Output feature vector
     * @param useEmb Whether to use embedding (true) or embeddingless (false) features.
     */
    private void addEmbeddingFeatures(LocalObservations local, List<FeatureVector> fvs) {
        //  - Per word, we have various features, such as whether a word is in between
        //    the entities or on the dependency path between them.
        //    - For each of the above features, if the feature fires, set the
        //      values to the embedding for that word.
        //    - For each of the above features, if the feature fires, set the
        //      values to the embedding for the word's super sense tag.
        NerMention m1 = local.getNe1();
        NerMention m2 = local.getNe2();
        Span m1span = m1.getSpan();
        Span m2span = m2.getSpan();
        
        FeaturizedSentence fsent = new FeaturizedSentence(sent, null);
        
        String ne1 = m1.getEntityType();
        String ne2 = m2.getEntityType();
        String sne1 = m1.getEntitySubType();
        String sne2 = m2.getEntitySubType();
        String ne1ne2 = ne1 + ne2;
                
        switch (prm.embFeatType) {
        case FULL:            
            //     - chunk_head
            //     - chunk_head+ne1
            //     - chunk_head+ne2
            //     - chunk_head+ne1+ne2
            Pair<List<LabeledSpan>, IntArrayList> chunkPair = RelObsFe.getSpansFromBIO(sent.getChunks(), true);
            List<LabeledSpan> chunks = chunkPair.get1();
            IntArrayList tokIdxToChunkIdx = chunkPair.get2();
            int c1 = tokIdxToChunkIdx.get(m1.getHead());
            int c2 = tokIdxToChunkIdx.get(m2.getHead());
            int[] chunkHeads = RelObsFe.getHeadsOfSpans(chunks, sent.getParents());
            for (int b=c1+1; b<=c2-1; b++) {
                int i = chunkHeads[b];
                addEmbFeat("chunk_head", i, fvs);
                addEmbFeat("chunk_head-t1"+ne1, i, fvs);
                addEmbFeat("chunk_head-t2"+ne2, i, fvs);
                addEmbFeat("chunk_head-t1t2"+ne1ne2, i, fvs);
            }
            
        case HEAD_TYPE_LOC_ST:
            //     - ne1_head+sne1
            //     - ne1_head+sne2
            addEmbFeat("ne1_head-st1"+sne1,    m1.getHead(), fvs);
            addEmbFeat("ne1_head-st2"+sne2,    m1.getHead(), fvs);
            
            //     - ne2_head+sne1
            //     - ne2_head+sne2
            addEmbFeat("ne2_head-st1"+sne1,    m2.getHead(), fvs);
            addEmbFeat("ne2_head-st2"+sne2,    m2.getHead(), fvs);
                        
        case HEAD_TYPE_LOC:
            //     - in_between: is the word in between entities
            //     - in_between+ne1 if in_between = T: ne1 is the entity type
            //     - in_between+ne2 if in_between = T
            //     - in_between+ne1+ne2 if in_between = T
            Span btwn = new Span(m1.getHead()+1, m2.getHead());
            for (int i=btwn.start(); i<btwn.end(); i++) {
                addEmbFeat("in_between", i, fvs);
                addEmbFeat("in_between-t1"+ne1, i, fvs);
                addEmbFeat("in_between-t2"+ne2, i, fvs);
                addEmbFeat("in_between-t1t2"+ne1ne2, i, fvs);
            }
    
            //     - on_path
            //     - on_path+ne1 if on_path = T
            //     - on_path+ne2 if on_path = T
            //     - on_path+ne1+ne2 if on_path = T
            FeaturizedTokenPair ftp = fsent.getFeatTokPair(m1.getHead(), m2.getHead());
            if (sent.getParents() != null) {
                List<Pair<Integer, ParentsArray.Dir>> depPath = ftp.getDependencyPath();
                if (depPath != null) {
                    for (Pair<Integer,ParentsArray.Dir> pair : depPath) {
                        int i = pair.get1();
                        addEmbFeat("on_path", i, fvs);
                        addEmbFeat("on_path-t1"+ne1, i, fvs);
                        addEmbFeat("on_path-t2"+ne2, i, fvs);
                        addEmbFeat("on_path-t1t2"+ne1ne2, i, fvs);
                    }
                } else {
                    log.trace("No dependency path between mention heads");
                }
            } else {
                log.trace("No dependency tree for sentence");
            }
            
            //     - -1_ne1: immediately to the left of the ne1 head
            //     - +1_ne1: immediately to the right of the ne1 head
            //     - -2_ne1: two to the left of the ne1 head
            //     - +2_ne1: two to the right of the ne1 head
            addEmbFeat("-1_ne1", m1.getHead()-1, fvs);
            addEmbFeat("+1_ne1", m1.getHead()+1, fvs);
            addEmbFeat("-2_ne1", m1.getHead()-2, fvs);
            addEmbFeat("+2_ne1", m1.getHead()+2, fvs);
            
            //     - -1_ne2: immediately to the left of the ne2 head
            //     - +1_ne2: immediately to the right of the ne2 head
            //     - -2_ne2: two to the left of the ne2 head
            //     - +2_ne2: two to the right of the ne2 head
            addEmbFeat("-1_ne2", m2.getHead()-1, fvs);
            addEmbFeat("+1_ne2", m2.getHead()+1, fvs);
            addEmbFeat("-2_ne2", m2.getHead()-2, fvs);
            addEmbFeat("+2_ne2", m2.getHead()+2, fvs);
                    
        case HEAD_TYPE:
            //     - ne1_head+ne1
            //     - ne1_head+ne2
            //     - ne1_head+ne1+ne2
            addEmbFeat("ne1_head-t1"+ne1,    m1.getHead(), fvs);
            addEmbFeat("ne1_head-t2"+ne2,    m1.getHead(), fvs);
            addEmbFeat("ne1_head-t1t2"+ne1ne2, m1.getHead(), fvs);
            
            //     - ne2_head+ne1
            //     - ne2_head+ne2
            //     - ne2_head+ne1+ne2
            addEmbFeat("ne2_head-t1"+ne1,    m2.getHead(), fvs);
            addEmbFeat("ne2_head-t2"+ne2,    m2.getHead(), fvs);
            addEmbFeat("ne2_head-t1t2"+ne1ne2, m2.getHead(), fvs);
            
        case HEAD_ONLY:
            //     - ne1_head: true if is the head of the first entity
            addEmbFeat("ne1_head",        m1.getHead(), fvs);
            //     - ne2_head: true if is the head of the second entity
            addEmbFeat("ne2_head",        m2.getHead(), fvs);
        }
    }
    
    private void addEmbFeat(String fname, int i, List<FeatureVector> fvs) {
        if (i < 0 || sent.size() <= i) {
            return;
        }
        FeatureVector fv = fvs.get(i);
        int fidx = alphabet.lookupIndex(fname);
        if (fidx != -1) {
            fv.add(fidx, 1.0);
        }
    }
    
}
