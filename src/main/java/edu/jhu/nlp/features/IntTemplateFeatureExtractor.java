package edu.jhu.nlp.features;

import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.CorpusStatistics;
import edu.jhu.nlp.data.simple.AlphabetStore;
import edu.jhu.nlp.data.simple.IntAnnoSentence;
import edu.jhu.nlp.features.TemplateLanguage.EdgeProperty;
import edu.jhu.nlp.features.TemplateLanguage.FeatTemplate;
import edu.jhu.nlp.features.TemplateLanguage.FeatTemplate0;
import edu.jhu.nlp.features.TemplateLanguage.FeatTemplate1;
import edu.jhu.nlp.features.TemplateLanguage.FeatTemplate2;
import edu.jhu.nlp.features.TemplateLanguage.FeatTemplate3;
import edu.jhu.nlp.features.TemplateLanguage.FeatTemplate4;
import edu.jhu.nlp.features.TemplateLanguage.JoinTemplate;
import edu.jhu.nlp.features.TemplateLanguage.ListModifier;
import edu.jhu.nlp.features.TemplateLanguage.OtherFeat;
import edu.jhu.nlp.features.TemplateLanguage.Position;
import edu.jhu.nlp.features.TemplateLanguage.PositionList;
import edu.jhu.nlp.features.TemplateLanguage.PositionModifier;
import edu.jhu.nlp.features.TemplateLanguage.RulePiece;
import edu.jhu.nlp.features.TemplateLanguage.SymbolProperty;
import edu.jhu.nlp.features.TemplateLanguage.TokPropList;
import edu.jhu.nlp.features.TemplateLanguage.TokProperty;
import edu.jhu.pacaya.parse.cky.Rule;
import edu.jhu.pacaya.parse.dep.ParentsArray;
import edu.jhu.pacaya.util.hash.MurmurHash;
import edu.jhu.prim.Primitives;
import edu.jhu.prim.list.IntArrayList;
import edu.jhu.prim.list.ShortArrayList;
import edu.jhu.prim.set.IntHashSet;
import edu.jhu.prim.tuple.Pair;

/**
 * Defines a feature template extractor for templates based on a 'little
 * language'.
 * 
 * @author mgormley
 * @author mmitchell
 */
public class IntTemplateFeatureExtractor {
   
    private static final Logger log = LoggerFactory.getLogger(IntTemplateFeatureExtractor.class);

    private final IntAnnoSentence isent;
    private final FeaturizedSentence fSent;

    /**
     * This constructor is preferred as it allows the FeaturizedSentence to
     * share work across different feature extractors.
     */
    public IntTemplateFeatureExtractor(IntAnnoSentence isent, CorpusStatistics cs) {        
        this.isent = isent;
        this.fSent = new FeaturizedSentence(isent.getAnnoSentence(), cs);
    }
    
    /** Adds features for a list of feature templates. */
    public void addFeatures(List<FeatTemplate> tpls, LocalObservations local, IntArrayList feats) {
        for (FeatTemplate tpl : tpls) {
            addFeatures(tpl, local, feats);
        }
    }
    
    /** Adds features for a single feature template. */
    public void addFeatures(FeatTemplate tpl, LocalObservations local, IntArrayList feats) {
        if (tpl instanceof FeatTemplate1) {
            addTokenFeature((FeatTemplate1) tpl, local, feats);
        } else if (tpl instanceof FeatTemplate2) {
            addTokenFeatures((FeatTemplate2) tpl, local, feats);            
        } else if (tpl instanceof FeatTemplate3) {
            addListFeature((FeatTemplate3) tpl, local, feats);      
        } else if (tpl instanceof FeatTemplate4) {
            addRuleFeature((FeatTemplate4) tpl, local, feats);
        } else if (tpl instanceof FeatTemplate0) {
            addOtherFeature((FeatTemplate0) tpl, local, feats);
        } else if (tpl instanceof JoinTemplate) {
            addJoinFeature((JoinTemplate) tpl, local, feats);
        } else {
            throw new IllegalStateException("Feature not supported: " + tpl);
        }
    }
    
    /**
     * For n-gram feature templates of the form:
     *     w(p) + bc0(-1(c))
     *     t(p) + t(c)
     *     t(p) + t(c) + w(p)
     * @param tpl Structured feature template.
     * @param local Local observations.
     * @param feats The feature list to which this will be added.
     */
    protected void addJoinFeature(JoinTemplate joinTpl, LocalObservations local, IntArrayList feats) {
        IntArrayList joined = new IntArrayList();
        addFeatures(joinTpl.tpls[0], local, joined);
        for (int i=1; i<joinTpl.tpls.length; i++) {
            IntArrayList tmpFeats = new IntArrayList();
            if (joined.size() == 0) {
                // Short circuit since we'll never create any features.
                return;
            }
            addFeatures(joinTpl.tpls[i], local, tmpFeats);
            joined = joinIntoBigrams(joined, tmpFeats);
        }
        feats.add(joined);
    }

    private IntArrayList joinIntoBigrams(IntArrayList feats1, IntArrayList feats2) {
        IntArrayList joined = new IntArrayList();
        for (int i=0; i<feats1.size(); i++) {
            int f1 = feats1.get(i);
            for (int j=0; j<feats2.size(); j++) {
                int f2 = feats2.get(j);
                joined.add(toFeat(f1, f2));
            }
        }
        return joined;
    }

    /**
     * Adds feature templates of the form: 
     *     tag(ruleP)
     *     bTag(ruleLc)
     * @param tpl Structured feature template.
     * @param local Local observations.
     * @param feats The feature list to which this will be added.
     */
    protected void addRuleFeature(FeatTemplate4 tpl, LocalObservations local, IntArrayList feats) {
        RulePiece piece = tpl.piece; SymbolProperty prop = tpl.prop;
        Rule rule = local.getRule();        
        
        // Get a symbol from the rule.
        int symbol;
        switch (piece) {
        case PARENT: symbol = rule.getParent(); break;
        case LEFT_CHILD: symbol = rule.getLeftChild(); break;
        case RIGHT_CHILD: symbol = rule.getRightChild(); break;
        default: throw new IllegalStateException();
        }
        
        // Get a property of that symbol.
        int val;
        switch (prop) {
        case TAG: val = symbol; break;        
        default: throw new IllegalStateException();
        }
        
        // Create the feature.
        if (val != -1) {
            feats.add(toFeat(tpl.getId(), val));
        }
    }

    /**
     * Adds features of the form: 
     *     bc1(p)
     *     dr(head(c))
     *     bc0(first(t, NOUN, path(p, root)))
     * @param tpl Structured feature template.
     * @param local Local observations.
     * @param feats The feature list to which this will be added.
     */
    protected void addTokenFeature(FeatTemplate1 tpl, LocalObservations local, IntArrayList feats) {
        Position pos = tpl.pos; PositionModifier mod = tpl.mod; TokProperty prop = tpl.prop;
        int idx = getIndexOfPosition(local, pos);
        idx = getModifiedPosition(mod, idx);
        int val = getTokProp(prop, idx);
        feats.add(toFeat(tpl.getId(), val));
    }

    private int getIndexOfPosition(LocalObservations local, Position pos) {
        switch (pos) {
        case PARENT: return local.getPidx();
        case CHILD: return local.getCidx();
        case MODIFIER: return local.getMidx();
        case RULE_START: return local.getRStartIdx();
        case RULE_MID: return local.getRMidIdx();
        case RULE_END: return local.getREndIdx();
        case M_1_START: return local.getNe1().getSpan().start();
        case M_1_END: return local.getNe1().getSpan().end();
        case M_1_HEAD: return local.getNe1().getHead();
        case M_2_START: return local.getNe2().getSpan().start();
        case M_2_END: return local.getNe2().getSpan().end();
        case M_2_HEAD: return local.getNe2().getHead();
        default: throw new IllegalStateException();
        }
    }
    
    /** Same as above except that it permits properties of the token which expand to multiple strings. */
    protected void addTokenFeatures(FeatTemplate2 tpl, LocalObservations local, IntArrayList feats) {
        Position pos = tpl.pos; PositionModifier mod = tpl.mod; TokPropList prop = tpl.prop;
        int idx = getIndexOfPosition(local, pos);
        idx = getModifiedPosition(mod, idx);
        ShortArrayList vals = getTokPropList(prop, idx);
        for (int i=0; i<vals.size(); i++) {
            int val = vals.get(i);
            feats.add(toFeat(tpl.getId(), val));
        }
    }

    private final static int NO_PATH_ID = "NO_PATH".hashCode();

    /**
     * Gets features of the form: 
     *    noDup(bc0+dir(path(lca(p,c),root)))
     *    seq(bc0(children(p)))
     *    noDup(t(line(p,c)))
     *    
     * @param tpl Structured feature template.
     * @param local Local observations.
     * @param feats The feature list to which this will be added.
     */
    protected void addListFeature(FeatTemplate3 tpl, LocalObservations local, IntArrayList feats) {
        PositionList pl = tpl.pl; TokProperty prop = tpl.prop; EdgeProperty eprop = tpl.eprop; ListModifier lmod = tpl.lmod;
        
        if (prop == null && eprop == null) {
            throw new IllegalStateException("Feature template extracts nothing. One of prop and eprop must be non-null.");
        }
        IntArrayList vals;
        switch (pl) {
        case CHILDREN_P: case NO_FAR_CHILDREN_P: case CHILDREN_C: case NO_FAR_CHILDREN_C: case LINE_P_C: case LINE_RI_RK: case BTWN_P_C:
            if (eprop != null) {
                throw new IllegalStateException("EdgeProperty " + eprop + " is only supported on paths. Offending template: " + tpl);
            } else if (prop == null) {
                throw new IllegalStateException("TokProperty must be non-null for position lists.");
            }
            IntArrayList posList = getPositionList(pl, local);
            vals = getTokPropsForList(prop, posList);
            listAndPathHelper(vals, lmod, tpl, feats);
            return;
        case PATH_P_C: case PATH_C_LCA: case PATH_P_LCA: case PATH_LCA_ROOT: 
            List<Pair<Integer, ParentsArray.Dir>> path = getPath(pl, local);
            if (path != null) {
                vals = getTokPropsForPath(prop, eprop, path);
                listAndPathHelper(vals, lmod, tpl, feats);
            } else {
                // No path.
                feats.add(toFeat(tpl.getId(), AlphabetStore.TOK_UNK_INT));
            }
            return;
        default:
            throw new IllegalStateException();
        }
    }

    private void listAndPathHelper(IntArrayList vals, ListModifier lmod, FeatTemplate3 tpl, IntArrayList feats) {
        int feat;
        if (lmod == ListModifier.UNIGRAM) {
            for (int i=0; i<vals.size(); i++) {
                int v = vals.get(i);
                feat = toFeat(tpl.getId(), v);
                feats.add(feat);
            }
        } else if (lmod == ListModifier.BIGRAM) {
            for (int i = -1; i < vals.size() - 1; i++) {
                feat = toFeat(tpl.getId(), safeGet(vals, i), safeGet(vals, i+1));
                feats.add(feat);
            }
        } else if (lmod == ListModifier.TRIGRAM) {
            for (int i = -2; i < vals.size() - 2; i++) {
                feat = toFeat(tpl.getId(), safeGet(vals, i), safeGet(vals, i+1), safeGet(vals, i+2));
                feats.add(feat);
            }
        } else {
            IntArrayList modList = getModifiedList(lmod, vals);
            feat = toFeat(tpl.getId(), modList);
            feats.add(feat);
        }
    }

    private int safeGet(IntArrayList vals, int i) {
        if (i < 0) {
            return AlphabetStore.TOK_START_INT;
        } else if(vals.size() <= i) {
            return AlphabetStore.TOK_END_INT;
        } else {
            return vals.get(i);
        }
    }

    /**
     * Gets special features.
     *
     * @param tpl Structured feature template.
     * @param local Local observations.
     * @param feats The feature list to which this will be added.
     */
    protected void addOtherFeature(FeatTemplate0 tpl, LocalObservations local, IntArrayList feats) {
        OtherFeat template = tpl.feat;  
        switch (template) {
        case PATH_GRAMS:
            List<Pair<Integer,ParentsArray.Dir>> path = getPath(PositionList.PATH_P_C, local);  
            if (path != null) {
                addPathGrams(tpl, path, feats);
            } else {
                feats.add(toFeat(tpl.getId(), NO_PATH_ID));
            }
            return;
        default:  
            int val = getOtherFeatSingleton(tpl.feat, local);
            feats.add(toFeat(tpl.getId(), val));
            return;
        }
    }

    // TODO: This is a lot of logic...and should probably live elsewhere.
    private void addPathGrams(FeatTemplate0 tpl, List<Pair<Integer, ParentsArray.Dir>> path, IntArrayList feats) {
        // For each path n-gram, for n in {1,2,3}:
        for (int n=1; n<=3; n++) {
            for (int start = 0; start <= path.size() - n; start++) {
                int end = start + n;
                List<Pair<Integer,ParentsArray.Dir>> ngram = path.subList(start, end);
                // For each pattern of length n, comprised of WORD and POS.
                TokProperty[] props = new TokProperty[] { TokProperty.WORD, TokProperty.POS };
                int max = (int) Math.pow(2, n);
                for (int pattern = 0; pattern < max; pattern++) {
                    // Create the feature for this pattern.
                    IntArrayList vals = new IntArrayList(n);
                    for (int i=0; i<n; i++) {
                        // Get the appropriate type for this pattern:
                        // ((pattern>>>i) & 1) is 1 if the i'th bit is one and 0 otherwise.
                        TokProperty prop = props[(pattern>>>i) & 1];
                        vals.add(getTokPropsForPath(prop, null, ngram.subList(i, i+1)));
                    }
                    // Add the feature for this pattern.
                    feats.add(toFeat(tpl.getId(), vals));
                }
            }
        }
    }
    
    private int getOtherFeatSingleton(OtherFeat template, LocalObservations local) {
        int pidx = local.getPidx();
        int cidx = local.getCidx();
        FeaturizedTokenPair pair = getFeatTokPair(pidx, cidx);
        int[] parents;
        switch (template) {
        case DISTANCE:
            return Math.abs(pidx - cidx);
        case RELATIVE:
            return pair.getRelativePosition().ordinal();
        case UNDIR_EDGE:
            parents = isent.getAnnoSentence().getParents();
            return (parents[cidx] == pidx || (pidx != -1 && parents[pidx] == cidx)) ? 1 : 0; 
        case DIR_EDGE:
            parents = isent.getAnnoSentence().getParents();
            return (parents[cidx] == pidx) ? 1 : 0; 
        case GENEOLOGY:
            return pair.getGeneologicalRelation().ordinal();
        case CONTINUITY:
            return pair.getCountOfNonConsecutivesInPath();
        case PATH_LEN:
            List<Pair<Integer, ParentsArray.Dir>> depPath = pair.getDependencyPath();
            int pathLen = depPath == null ? 0 : depPath.size();
            return binInt(pathLen, 0, 2, 5, 10, 20, 30, 40);
        case SENT_LEN:            
            return binInt(isent.size(), 0, 2, 5, 10, 20, 30, 40);
        case RULE_IS_UNARY:
            return local.getRule().isUnary() ? 1 : 0;
        default:
            throw new IllegalStateException();
        }
    }

    public static int binInt(int size, int...bins) {
        for (int i=bins.length-1; i >= 0; i--) {
            if (size >= bins[i]) {
                return bins[i];
            }
        }
        return Integer.MIN_VALUE;
    }

    private IntArrayList getModifiedList(ListModifier lmod, IntArrayList props) {
        switch (lmod) {
        case SEQ:
            return props;
        case BAG:
            return new IntArrayList(new IntHashSet(props).toNativeArray());
        case NO_DUP:
            props.uniq();
            return props;
        default:
            throw new IllegalStateException();
        }
    }

    private IntArrayList getPositionList(PositionList pl, LocalObservations local) {              
        FeaturizedToken tok;
        FeaturizedTokenPair pair;
        switch (pl) {
        case CHILDREN_P: 
            tok = getFeatTok(local.getPidx());
            return tok.getChildren();
        case NO_FAR_CHILDREN_P: 
            tok = getFeatTok(local.getPidx());
            return tok.getNoFarChildren();
        case CHILDREN_C: 
            tok = getFeatTok(local.getCidx());
            return tok.getChildren();
        case NO_FAR_CHILDREN_C: 
            tok = getFeatTok(local.getCidx());
            return tok.getNoFarChildren();
        case LINE_P_C: 
            pair = getFeatTokPair(local.getPidx(), local.getCidx());
            return pair.getLinePath();
        case LINE_RI_RK: 
            pair = getFeatTokPair(local.getRStartIdx(), local.getREndIdx());
            return pair.getLinePath();
        case BTWN_P_C:
            pair = getFeatTokPair(local.getPidx(), local.getCidx());
            return pair.getBtwnPath();
        default:
            throw new IllegalStateException();
        }
    }
    
    /** Gets the desired path or null if it doesn't exist. */
    private List<Pair<Integer, ParentsArray.Dir>> getPath(PositionList pl, LocalObservations local) {        
        FeaturizedTokenPair pair = getFeatTokPair(local.getPidx(), local.getCidx());
        switch (pl) {
        case PATH_P_C:
            return pair.getDependencyPath();
        case PATH_C_LCA:
            return pair.getDpPathArg();
        case PATH_P_LCA:
            return pair.getDpPathPred();
        case PATH_LCA_ROOT:
            return pair.getDpPathShare();
        default:
            throw new IllegalStateException();
        }
    }
    
    private int getModifiedPosition(PositionModifier mod, int idx) {
        FeaturizedToken tok = null;
        switch (mod) {
            // --------------------- Word ---------------------  
        case IDENTITY:
            return idx;
        case BEFORE1:
            return idx - 1;
        case BEFORE2:
            return idx - 2;
        case AFTER1:
            return idx + 1;
        case AFTER2:
            return idx + 2;
            // --------------------- DepTree ---------------------  
        default:
            tok = getFeatTok(idx);
            switch (mod) {
            case HEAD:
                return tok.getParent();
            case LNS:
                return tok.getNearLeftSibling();
            case RNS:
                return tok.getNearRightSibling();
            case LMC:
                return tok.getFarLeftChild();
            case RMC:
                return tok.getFarRightChild();
            case LNC:
                return tok.getNearLeftChild();
            case RNC:
                return tok.getNearRightChild();     
            case LOW_SV:
                return tok.getLowSupportVerb();
            case LOW_SN:
                return tok.getLowSupportNoun();
            case HIGH_SV:
                return tok.getHighSupportVerb();
            case HIGH_SN:
                return tok.getHighSupportNoun();
            default:
                throw new IllegalStateException();
            }
        }
    }
    
    public IntArrayList getTokPropsForPath(TokProperty prop, EdgeProperty eprop, List<Pair<Integer,ParentsArray.Dir>> path) {
        IntArrayList props = new IntArrayList(path.size());
        for (int i=0; i<path.size(); i++) {
            Pair<Integer,ParentsArray.Dir> edge = path.get(i);
            if (prop != null) {
                int val = getTokProp(prop, edge.get1());
                props.add(val);
            }
            if (eprop != null && i < path.size() - 1) {
                switch (eprop) {
                case DIR: props.add(edge.get2().ordinal()); break;
                case EDGEREL:
                    int idx1 = path.get(i).get1();
                    int idx2 = path.get(i+1).get1();
                    ParentsArray.Dir d = path.get(i).get2();
                    int idx = (d == ParentsArray.Dir.UP) ? idx1 : idx2;
                    props.add(getTokProp(TokProperty.DEPREL, idx));                    
                    break;
                default: throw new IllegalStateException();
                }
            }
        }
        return props;
    }

    private IntArrayList getTokPropsForList(TokProperty prop, IntArrayList posList) {
        IntArrayList props = new IntArrayList(posList.size());
        for (int i=0; i<posList.size(); i++) {
            int idx = posList.get(i);
            int val = getTokProp(prop, idx);
            props.add(val);
        }
        return props;
    }

    // package private for testing.
    ShortArrayList getTokPropList(TokPropList prop, int idx) {
        if (idx < 0) { return new ShortArrayList(new short[]{AlphabetStore.TOK_START_INT}); }
        if (idx >= isent.size()) { return new ShortArrayList(new short[]{AlphabetStore.TOK_END_INT}); }
        switch (prop) {
        case EACH_MORPHO:
            ShortArrayList vals = isent.getFeats(idx);
            if (vals.size() == 0) { return new ShortArrayList(new short[]{AlphabetStore.TOK_UNK_INT}); }
            else { return vals; }
        default:
            throw new IllegalStateException();
        }
    }
    
    /**
     * @return The property or null if the property is not included.
     */
    // package private for testing.
    int getTokProp(TokProperty prop, int idx) {
        if (idx < 0) { return AlphabetStore.TOK_START_INT; }
        if (idx >= isent.size()) { return AlphabetStore.TOK_END_INT; }
        switch (prop) {
        case WORD: return isent.getWord(idx);
        case INDEX: return idx;
        case LC: return isent.getLcWord(idx);
        case CAPITALIZED: return isent.isCapitalized(idx) ? 1 : 0;
        case WORD_TOP_N:
            short word = isent.getWord(idx);
            AlphabetStore store = isent.getStore();
            if (store.getWordTypeCount(word) > store.getWordTopNCutoff()) {
                return word;
            } else {
                return AlphabetStore.TOK_UNK_INT;
            }
        case CHPRE1: return isent.getPrefix(idx, 1);
        case CHPRE2: return isent.getPrefix(idx, 2);
        case CHPRE3: return isent.getPrefix(idx, 3);
        case CHPRE4: return isent.getPrefix(idx, 4);
        case CHPRE5: return isent.getPrefix(idx, 5);
        case CHSUF1: return isent.getSuffix(idx, 1);
        case CHSUF2: return isent.getSuffix(idx, 2);
        case CHSUF3: return isent.getSuffix(idx, 3);
        case CHSUF4: return isent.getSuffix(idx, 4);
        case CHSUF5: return isent.getSuffix(idx, 5);
        case LEMMA: return isent.getLemma(idx);
        case POS: return isent.getPosTag(idx);
        case CPOS: return isent.getCposTag(idx);
        case BC0: return isent.getClusterPrefix(idx, 5);
        case BC1: return isent.getCluster(idx);
        case DEPREL: return isent.getDeprel(idx);
        case MORPHO: return toFeat(isent.getFeats(idx));
        case MORPHO1: return safeGet(isent.getFeats(idx), 0);
        case MORPHO2: return safeGet(isent.getFeats(idx), 1);
        case MORPHO3: return safeGet(isent.getFeats(idx), 2);
        case UNK: throw new RuntimeException("not implemented");
        default:
            throw new IllegalStateException();
        }
    }

    private int toFeat(IntArrayList feats) {
        return MurmurHash.hash32(feats.getInternalElements(), feats.size());
    }
    
    private int toFeat(ShortArrayList feats) {
        return MurmurHash.hash32(feats.getInternalElements(), feats.size());
    }

    private short safeGet(ShortArrayList feats, int i) {
        if (i < feats.size()) { return feats.get(i); }
        else { return AlphabetStore.TOK_UNK_INT; }
    }
    
    private FeaturizedToken getFeatTok(int idx) {
        return fSent.getFeatTok(idx);
    }
    
    private FeaturizedTokenPair getFeatTokPair(int pidx, int cidx) {
        return fSent.getFeatTokPair(pidx, cidx);
    }
    
    private int toFeat(int f1, IntArrayList fs) {
        int fsHash = MurmurHash.hash32(fs.getInternalElements(), fs.size());
        return toFeat(f1, fsHash);
    }
    
    private int toFeat(int f1, int f2, int f3, int f4) {
        int f123 = toFeat(f1, f2, f3);
        return toFeat(f123, f4);
    }
    
    private int toFeat(int f1, int f2, int f3) {
        int f12 = toFeat(f1, f2);
        return toFeat(f12, f3);
    }

    private static final long INT_MAX = Primitives.LONG_MAX_UINT;

    private int toFeat(int f1, int f2) {
        long feat =  (f1 & INT_MAX) | ((f2 & INT_MAX) << 32);
        return MurmurHash.hash32(feat);
    }
    
}
