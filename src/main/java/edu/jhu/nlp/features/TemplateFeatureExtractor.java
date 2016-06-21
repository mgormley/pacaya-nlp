package edu.jhu.nlp.features;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.TreeSet;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.CorpusStatistics;
import edu.jhu.nlp.data.simple.AnnoSentence;
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
import edu.jhu.pacaya.util.collections.QLists;
import edu.jhu.prim.list.IntArrayList;
import edu.jhu.prim.tuple.Pair;

/**
 * Defines a feature template extractor for templates based on a 'little
 * language'.
 * 
 * @author mgormley
 * @author mmitchell
 */
public class TemplateFeatureExtractor {
   
    private static final Logger log = LoggerFactory.getLogger(TemplateFeatureExtractor.class);

    private final CorpusStatistics cs;
    private final AnnoSentence sent;
    private final FeaturizedSentence fSent; 

    /**
     * This constructor is preferred as it allows the FeaturizedSentence to
     * share work across different feature extractors.
     */
    public TemplateFeatureExtractor(FeaturizedSentence fSent, CorpusStatistics cs) {        
        this.cs = cs;
        this.fSent = fSent;
        this.sent = fSent.getSent();
    }
    
    public TemplateFeatureExtractor(AnnoSentence sent, CorpusStatistics cs) {
        this.cs = cs;
        this.fSent = new FeaturizedSentence(sent, cs);
        this.sent = fSent.getSent();
    }
            
    /** Adds features for a list of feature templates. */
    public void addFeatures(List<FeatTemplate> tpls, LocalObservations local, List<String> feats) {
        for (FeatTemplate tpl : tpls) {
            addFeatures(tpl, local, feats);
        }
    }
    
    /** Adds features for a single feature template. */
    public void addFeatures(FeatTemplate tpl, LocalObservations local, List<String> feats) {
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
     *     p.w+c_{-1}.bc0
     *     p.t+c.t
     *     p.t+c.t+p.w
     * @param tpl Structured feature template.
     * @param local Local observations.
     * @param feats The feature list to which this will be added.
     */
    protected void addJoinFeature(JoinTemplate joinTpl, LocalObservations local, List<String> feats) {
        ArrayList<String> joined = new ArrayList<String>();
        addFeatures(joinTpl.tpls[0], local, joined);
        for (int i=1; i<joinTpl.tpls.length; i++) {
            ArrayList<String> tmpFeats = new ArrayList<String>();
            if (joined.size() == 0) {
                // Short circuit since we'll never create any features.
                return;
            }
            addFeatures(joinTpl.tpls[i], local, tmpFeats);
            joined = joinIntoBigrams(joined, tmpFeats);
        }
        feats.addAll(joined);
    }

    private ArrayList<String> joinIntoBigrams(ArrayList<String> feats1, ArrayList<String> feats2) {
        ArrayList<String> joined = new ArrayList<String>();
        for (String f1 : feats1) {
            for (String f2 : feats2) {
                joined.add(toFeat(f1, f2));
            }
        }
        return joined;
    }

    /**
     * Adds feature templates of the form: 
     *     ruleP.tag
     *     ruleLc.bTag
     * @param tpl Structured feature template.
     * @param local Local observations.
     * @param feats The feature list to which this will be added.
     */
    protected void addRuleFeature(FeatTemplate4 tpl, LocalObservations local, List<String> feats) {
        RulePiece piece = tpl.piece; SymbolProperty prop = tpl.prop;
        Rule rule = local.getRule();        
        
        // Get a symbol from the rule.
        String symbol;
        switch (piece) {
        case PARENT: symbol = rule.getParentLabel(); break;
        case LEFT_CHILD: symbol = rule.getLeftChildLabel(); break;
        case RIGHT_CHILD: symbol = rule.getRightChildLabel(); break;
        default: throw new IllegalStateException();
        }
        
        // Get a property of that symbol.
        String val;
        switch (prop) {
        case TAG: val = symbol; break;        
        default: throw new IllegalStateException();
        }
        
        // Create the feature.
        if (val != null) {
            feats.add(toFeat(tpl.getName(), val));
        }
    }

    /**
     * Adds features of the form: 
     *     p.bc1
     *     c_{head}.dr
     *     first(t, NOUN, path(p, root)).bc0
     * @param tpl Structured feature template.
     * @param local Local observations.
     * @param feats The feature list to which this will be added.
     */
    protected void addTokenFeature(FeatTemplate1 tpl, LocalObservations local, List<String> feats) {
        Position pos = tpl.pos; PositionModifier mod = tpl.mod; TokProperty prop = tpl.prop;
        int idx = getIndexOfPosition(local, pos);
        idx = getModifiedPosition(mod, idx);
        String val = getTokProp(prop, idx);
        if (val != null) {
            feats.add(toFeat(tpl.getName(), val));
        }
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
    protected void addTokenFeatures(FeatTemplate2 tpl, LocalObservations local, List<String> feats) {
        Position pos = tpl.pos; PositionModifier mod = tpl.mod; TokPropList prop = tpl.prop;
        int idx = getIndexOfPosition(local, pos);
        idx = getModifiedPosition(mod, idx);
        List<String> vals = getTokPropList(prop, idx);
        for (String val : vals) {
            feats.add(toFeat(tpl.getName(), val));
        }
    }

    /**
     * Gets features of the form: 
     *    path(lca(p,c),root).bc0+dir.noDup
     *    children(p).bc0.seq
     *    line(p,c).t.noDup
     *    
     * @param tpl Structured feature template.
     * @param local Local observations.
     * @param feats The feature list to which this will be added.
     */
    protected void addListFeature(FeatTemplate3 tpl, LocalObservations local, List<String> feats) {
        PositionList pl = tpl.pl; TokProperty prop = tpl.prop; EdgeProperty eprop = tpl.eprop; ListModifier lmod = tpl.lmod;
        
        if (prop == null && eprop == null) {
            throw new IllegalStateException("Feature template extracts nothing. One of prop and eprop must be non-null.");
        }
        
        List<String> vals;
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
                feats.add(toFeat(tpl.getName(), "NO_PATH"));
            }
            return;
        default:
            throw new IllegalStateException();
        }
    }

    private void listAndPathHelper(List<String> vals, ListModifier lmod, FeatTemplate3 tpl, List<String> feats) {
        String feat;
        if (lmod == ListModifier.UNIGRAM) {
            for (String v : vals) {
                feat = toFeat(tpl.getName(), v);
                feats.add(feat);
            }
        } else if (lmod == ListModifier.BIGRAM) {
            for (int i = -1; i < vals.size() - 1; i++) {
                feat = toFeat(tpl.getName(), safeGet(vals, i), safeGet(vals, i+1));
                feats.add(feat);
            }
        } else if (lmod == ListModifier.TRIGRAM) {
            for (int i = -2; i < vals.size() - 2; i++) {
                feat = toFeat(tpl.getName(), safeGet(vals, i), safeGet(vals, i+1), safeGet(vals, i+2));
                feats.add(feat);
            }
        } else {
            Collection<String> modList = getModifiedList(lmod, vals);
            feat = toFeat(tpl.getName(), modList);
            feats.add(feat);
        }
    }

    private String safeGet(List<String> vals, int i) {
        if (i < 0) {
            return "BOS";
        } else if(vals.size() <= i) {
            return "EOS";
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
    protected void addOtherFeature(FeatTemplate0 tpl, LocalObservations local, List<String> feats) {
        OtherFeat template = tpl.feat;  
        switch (template) {
        case PATH_GRAMS:
            List<Pair<Integer,ParentsArray.Dir>> path = getPath(PositionList.PATH_P_C, local);  
            if (path != null) {
                addPathGrams(tpl, path, feats);
            } else {
                feats.add(toFeat(tpl.getName(), "NO_PATH"));
            }
            return;
        default:  
            String val = getOtherFeatSingleton(tpl.feat, local);
            feats.add(toFeat(tpl.getName(), val));
            return;
        }
    }

    // TODO: This is a lot of logic...and should probably live elsewhere.
    private void addPathGrams(FeatTemplate0 tpl, List<Pair<Integer, ParentsArray.Dir>> path, List<String> feats) {
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
                    List<String> vals = new ArrayList<String>(n);
                    for (int i=0; i<n; i++) {
                        // Get the appropriate type for this pattern:
                        // ((pattern>>>i) & 1) is 1 if the i'th bit is one and 0 otherwise.
                        TokProperty prop = props[(pattern>>>i) & 1];
                        vals.addAll(getTokPropsForPath(prop, null, ngram.subList(i, i+1)));
                    }
                    // Add the feature for this pattern.
                    feats.add(toFeat(tpl.getName(), vals));
                }
            }
        }
    }
    
    private String getOtherFeatSingleton(OtherFeat template, LocalObservations local) {
        int pidx = local.getPidx();
        int cidx = local.getCidx();
        FeaturizedTokenPair pair = getFeatTokPair(pidx, cidx);
        int[] parents;
        switch (template) {
        case DISTANCE:
            return Integer.toString(Math.abs(pidx - cidx));
        case RELATIVE:
            return pair.getRelativePosition().name();
        case UNDIR_EDGE:
            parents = fSent.getSent().getParents();
            return ((cidx != -1 && parents[cidx] == pidx) || (pidx != -1 && parents[pidx] == cidx)) ? "T" : "F"; 
        case DIR_EDGE:
            parents = fSent.getSent().getParents();
            return (cidx != -1 && parents[cidx] == pidx) ? "T" : "F"; 
        case GENEOLOGY:
            return pair.getGeneologicalRelation().name();
        case CONTINUITY:
            return Integer.toString(pair.getCountOfNonConsecutivesInPath());
        case PATH_LEN:            
            List<Pair<Integer, ParentsArray.Dir>> depPath = pair.getDependencyPath();
            int pathLen = depPath == null ? 0 : depPath.size();
            return Integer.toString(binInt(pathLen, 0, 2, 5, 10, 20, 30, 40));
        case SENT_LEN:            
            return Integer.toString(binInt(fSent.size(), 0, 2, 5, 10, 20, 30, 40));
        case RULE_IS_UNARY:
            return local.getRule().isUnary() ? "T" : "F";
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

    private <T> Collection<T> getModifiedList(ListModifier lmod, Collection<T> props) {
        switch (lmod) {
        case SEQ:
            return props;
        case BAG:
            return bag(props);
        case NO_DUP:
            return QLists.getUniq(props);
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
        if (mod != PositionModifier.IDENTITY && mod != PositionModifier.AFTER1 && mod != PositionModifier.BEFORE1) {
            tok = getFeatTok(idx);
        }
        
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
    
    public List<String> getTokPropsForPath(TokProperty prop, EdgeProperty eprop, List<Pair<Integer,ParentsArray.Dir>> path) {
        List<String> props = new ArrayList<String>(path.size());
        for (int i=0; i<path.size(); i++) {
            Pair<Integer,ParentsArray.Dir> edge = path.get(i);
            if (prop != null) {
                String val = getTokProp(prop, edge.get1());
                if (val != null) {
                    props.add(val);
                }
            }
            if (eprop != null && i < path.size() - 1) {
                switch (eprop) {
                case DIR: props.add(edge.get2().name()); break;
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

    private List<String> getTokPropsForList(TokProperty prop, IntArrayList posList) {
        List<String> props = new ArrayList<String>(posList.size());
        for (int i=0; i<posList.size(); i++) {
            int idx = posList.get(i);
            String val = getTokProp(prop, idx);
            if (val != null) {
                props.add(val);
            }
        }
        return props;
    }

    // package private for testing.
    List<String> getTokPropList(TokPropList prop, int idx) {
        FeaturizedToken tok = getFeatTok(idx);
        switch (prop) {
        case EACH_MORPHO:
            return tok.getFeat();
        default:
            throw new IllegalStateException();
        }
    }
    
    /**
     * @return The property or null if the property is not included.
     */
    // package private for testing.
    String getTokProp(TokProperty prop, int idx) {
        if (idx < 0) { return "BOS"; }
        if (idx >= fSent.size()) { return "EOS"; }
        FeaturizedToken tok = getFeatTok(idx);
        String form;
        switch (prop) {
        case WORD:
            return sent.getWord(idx);
        case INDEX:
            return Integer.toString(idx);
        case LC:
            //TODO: return tok.getFormLc();
            return sent.getWord(idx).toLowerCase();
        case CAPITALIZED:
            return tok.isCapatalized() ? "UC" : "LC";
        case WORD_TOP_N:
            form = sent.getWord(idx);
            if (cs.topNWords.contains(form)) {
                return form;
            } else {
                return null;
            }
        case CHPRE1: return prefix(idx, 1);
        case CHPRE2: return prefix(idx, 2);
        case CHPRE3: return prefix(idx, 3);
        case CHPRE4: return prefix(idx, 4);
        case CHPRE5: return prefix(idx, 5);
        case CHSUF1: return suffix(idx, 1);
        case CHSUF2: return suffix(idx, 2);
        case CHSUF3: return suffix(idx, 3);
        case CHSUF4: return suffix(idx, 4);
        case CHSUF5: return suffix(idx, 5);
        case LEMMA:
            return sent.getLemma(idx);
        case POS:
            return sent.getPosTag(idx);
        case CPOS:
            return sent.getCposTag(idx);
        case STRICT_POS:
            return sent.getStrictPosTag(idx).name();
        case BC0:
            String bc = sent.getCluster(idx);
            return bc.substring(0, Math.min(bc.length(), 5));
        case BC1:
            return sent.getCluster(idx);
        case DEPREL:
            return sent.getDeprel(idx);
        case MORPHO:
            return tok.getFeatStr();
        case MORPHO1:
            return tok.getFeat6().get(0);
        case MORPHO2:
            return tok.getFeat6().get(1);
        case MORPHO3:
            return tok.getFeat6().get(2);
        default:
            throw new IllegalStateException();
        }
    }

    private String prefix(int idx, int max) {
        String s = sent.getWord(idx);
        return s.substring(0, Math.min(s.length(), max));
    }
    
    private String suffix(int idx, int max) {
        String s = sent.getWord(idx);
        return s.substring(Math.max(0, s.length() - max), s.length());
    }
    
    protected static <T> Collection<T> bag(Collection<T> elements) {
        // bag, which removes all duplicated strings and sort the rest
        return new TreeSet<T>(elements);
    }
    
    private FeaturizedToken getFeatTok(int idx) {
        return fSent.getFeatTok(idx);
    }
    
    private FeaturizedTokenPair getFeatTokPair(int pidx, int cidx) {
        return fSent.getFeatTokPair(pidx, cidx);
    }
    
    private String toFeat(String... vals) {
        return org.apache.commons.lang3.StringUtils.join(vals, "_");
    }

    private String toFeat(String name, Collection<String> vals) {
        return name + "_" + org.apache.commons.lang3.StringUtils.join(vals, "_");
    }

    private String toFeat(String f1, String f2) {
        return f1 + "_" + f2;
    }

}
