package edu.jhu.nlp.tag;

import static edu.jhu.nlp.data.simple.AlphabetStore.TOK_END_INT;
import static edu.jhu.nlp.data.simple.AlphabetStore.TOK_START_INT;
import static edu.jhu.nlp.data.simple.AlphabetStore.TOK_WALL_INT;
import static edu.jhu.nlp.features.BitPacking.encodeFeatureB___;
import static edu.jhu.nlp.features.BitPacking.encodeFeatureSBBB;
import static edu.jhu.nlp.features.BitPacking.encodeFeatureSBB_;
import static edu.jhu.nlp.features.BitPacking.encodeFeatureSB__;
import static edu.jhu.nlp.features.BitPacking.encodeFeatureSSB_;
import static edu.jhu.nlp.features.BitPacking.encodeFeatureSS__;
import static edu.jhu.nlp.features.BitPacking.encodeFeatureS___;

import edu.jhu.nlp.data.simple.IntAnnoSentence;
import edu.jhu.nlp.depparse.BitshiftDepParseFeatures;
import edu.jhu.pacaya.gm.feat.FeatureVector;
import edu.jhu.pacaya.util.hash.MurmurHash;
import edu.jhu.prim.util.SafeCast;
import edu.jhu.prim.util.math.FastMath;

public class BitshiftTokenFeatures {

    /**
     * IDs for feature collections.
     * @author mgormley
     */
    public static class FeatureCollection {
        
        public static int MAX_VAL = 0xf; // 4 bits
        private static int templ = 0;

        protected static byte next() {
            byte b = SafeCast.safeIntToUnsignedByte(templ++);
            if (b > MAX_VAL) { 
                throw new IllegalStateException("Too many feature collections.");
            }
            return b;
        }
        
        public static boolean isValid(byte type) {
            return (0 <= type && type <= MAX_VAL);
        }
        
        public static final byte UNIGRAM = next();
        public static final byte BIGRAM = next();
        
    }
    
    /** 
     * Template IDs for ARC features in {@link BitshiftDepParseFeatures}.
     * 
     * In the names below, we have the following mapping:
     * h := main word (head) 
     * l := token to the left of following position
     * r := token to the right of following position
     * 
     * W := word
     * P := prefix (length specified separately)
     * S := suffix (length specified separately)
     */
    static class TokTs {
        
        private static int templ = 0;

        protected static byte next() {
            return SafeCast.safeIntToUnsignedByte(templ++);
        }
        
        public static final byte BIAS = next();
        public static final byte hFirst = next();
        public static final byte hCap = next();
        public static final byte hW = next();
        public static final byte lhW = next();
        public static final byte rhW = next();
        public static final byte llhW = next();
        public static final byte rrhW = next();
        public static final byte hPr = next();
        public static final byte lhPr = next();
        public static final byte rhPr = next();
        public static final byte llhPr = next();
        public static final byte rrhPr = next();
        public static final byte hSu = next();
        public static final byte lhSu = next();
        public static final byte rhSu = next();
        public static final byte llhSu = next();
        public static final byte rrhSu = next();

        public static final byte hP_lhP = next();
        public static final byte hP_lhP_llhP = next();
        public static final byte hP_rhP = next();
        public static final byte hP_rhP_rrhP = next();
        public static final byte hC_lhC = next();
        public static final byte hC_lhC_llhC = next();
        public static final byte hC_rhC_rrhC = next();
        public static final byte hC_rhC = next();

        public static final byte hP = next();
        public static final byte hL = next();
        public static final byte hW_hP = next();
        public static final byte hC = next();
        public static final byte hW_hC = next();
        
        //        public static byte lhP;
        //        public static byte lhC;
        //        public static byte lhL;
        //        public static byte lhW_lhP;
        //        public static byte lhW_lhC;
        //        public static byte rhP;
        //        public static byte rhC;
        //        public static byte rhL;
        //        public static byte rhW_rhP;
        //        public static byte rhW_rhC;
        //        public static byte llhP;
        //        public static byte llhC;
        //        public static byte llhL;
        //        public static byte llhW_llhP;
        //        public static byte llhW_llhC;
        //        public static byte rrhP;
        //        public static byte rrhC;
        //        public static byte rrhL;
        //        public static byte rrhW_rrhP;
        //        public static byte rrhW_rrhC;
    }
    
    public static void addUnigramFeatures(IntAnnoSentence sent, int head, FeatureVector feats, int mod, short tagConfig) {
        //addWordFeatures(sent, head, feats, mod, FeatureCollection.UNIGRAM, tagConfig);
        addRichTokenFeatures(sent, head, feats, mod, tagConfig, 2, false, false);
    }
    
    public static void addBigramFeatures(IntAnnoSentence sent, int head, FeatureVector feats, int mod, short tagConfig) {
        byte flags = FeatureCollection.BIGRAM; // 4 bits.
        addFeat(feats, mod, encodeFeatureS___(TokTs.BIAS, flags, tagConfig));
    }
    
    public static void addWordFeatures(IntAnnoSentence sent, int head, FeatureVector feats, int mod, byte featCol, short tagConfig) {
        int sentLen = sent.size();

        // Flags for the type of feature.
        byte flags = featCol; // 4 bits.

        // Positional features.
        byte hFirst = (byte) ((head == 0) ? 1 : 0);
        
        // Word and its context words.
        short hWord = (head < 0) ? TOK_WALL_INT : sent.getWord(head);
        short lhWord = (head-1 < 0) ? TOK_START_INT : sent.getWord(head-1);
        short rhWord = (head+1 >= sentLen) ? TOK_END_INT : sent.getWord(head+1);
        short llhWord = (head-2 < 0) ? TOK_START_INT : sent.getWord(head-2);
        short rrhWord = (head+2 >= sentLen) ? TOK_END_INT : sent.getWord(head+2);

        // Word properties.
        byte hCap = (head < 0) ? 0 : (byte)(sent.isCapitalized(head) ? 1 : 0); // 1-bit
        
        byte len;
        
        // Prefixes
        len = 1;
        short hPre1 = (head < 0) ? TOK_WALL_INT : sent.getPrefix(head, len);
        short lhPre1 = (head-1 < 0) ? TOK_START_INT : sent.getPrefix(head-1, len);
        short rhPre1 = (head+1 >= sentLen) ? TOK_END_INT : sent.getPrefix(head+1, len);
        short llhPre1 = (head-2 < 0) ? TOK_START_INT : sent.getPrefix(head-2, len);
        short rrhPre1 = (head+2 >= sentLen) ? TOK_END_INT : sent.getPrefix(head+2, len);

        // Suffixes
        len = 3;
        short hSuf3 = (head < 0) ? TOK_WALL_INT : sent.getSuffix(head, len);
        short lhSuf3 = (head-1 < 0) ? TOK_START_INT : sent.getSuffix(head-1, len);
        short rhSuf3 = (head+1 >= sentLen) ? TOK_END_INT : sent.getSuffix(head+1, len);
        short llhSuf3 = (head-2 < 0) ? TOK_START_INT : sent.getSuffix(head-2, len);
        short rrhSuf3 = (head+2 >= sentLen) ? TOK_END_INT : sent.getSuffix(head+2, len);
        
        // --------------------------------------------------------------------
        // Bias Feature.
        // --------------------------------------------------------------------        
        addFeat(feats, mod, encodeFeatureB___(TokTs.BIAS, flags, (byte)0));

        // --------------------------------------------------------------------
        // Unigram Features of the word and context words.
        // --------------------------------------------------------------------

        // Position.
        addFeat(feats, mod, encodeFeatureSB__(TokTs.hFirst, flags, tagConfig, hFirst));

        // Word.
        addFeat(feats, mod, encodeFeatureSS__(TokTs.hW, flags, tagConfig, hWord));
        addFeat(feats, mod, encodeFeatureSS__(TokTs.lhW, flags, tagConfig, lhWord)); 
        addFeat(feats, mod, encodeFeatureSS__(TokTs.rhW, flags, tagConfig, rhWord));
        addFeat(feats, mod, encodeFeatureSS__(TokTs.llhW, flags, tagConfig, llhWord));
        addFeat(feats, mod, encodeFeatureSS__(TokTs.rrhW, flags, tagConfig, rrhWord));
        
        // Word properties.
        addFeat(feats, mod, encodeFeatureSB__(TokTs.hCap, flags, tagConfig, hCap));
        
        // Prefix.
        len = 1;
        addFeat(feats, mod, encodeFeatureSSB_(TokTs.hPr, flags, tagConfig, hPre1, len));
        addFeat(feats, mod, encodeFeatureSSB_(TokTs.lhPr, flags, tagConfig, lhPre1, len)); 
        addFeat(feats, mod, encodeFeatureSSB_(TokTs.rhPr, flags, tagConfig, rhPre1, len));
        addFeat(feats, mod, encodeFeatureSSB_(TokTs.llhPr, flags, tagConfig, llhPre1, len));
        addFeat(feats, mod, encodeFeatureSSB_(TokTs.rrhPr, flags, tagConfig, rrhPre1, len));

        // Suffix.
        len = 3;
        addFeat(feats, mod, encodeFeatureSSB_(TokTs.hSu, flags, tagConfig, hSuf3, len));
        addFeat(feats, mod, encodeFeatureSSB_(TokTs.lhSu, flags, tagConfig, lhSuf3, len)); 
        addFeat(feats, mod, encodeFeatureSSB_(TokTs.rhSu, flags, tagConfig, rhSuf3, len));
        addFeat(feats, mod, encodeFeatureSSB_(TokTs.llhSu, flags, tagConfig, llhSuf3, len));
        addFeat(feats, mod, encodeFeatureSSB_(TokTs.rrhSu, flags, tagConfig, rrhSuf3, len));
    }
    

    /**
     * Token features based on word, lemma, POS tag, coarse POS tag.
     */
    public static void addRichTokenFeatures(final IntAnnoSentence sent, final int head, 
            final FeatureVector feats, final int mod,
            final short tagConfig,
            final int maxTokenContext,
            final boolean useLemmaFeats,
            final boolean useCoarseTags) {
        
        int sentLen = sent.size();

        // words/tags/lemmas.
        byte hPos = (head < 0) ? TOK_START_INT : sent.getPosTag(head);
        byte hCpos = 0;
        if (useCoarseTags) {
            hCpos = (head < 0) ? TOK_START_INT : sent.getCposTag(head);
        }

        // Surrounding words / POS tags. 
        // One token to the left (l) and right (r).
        //
        byte lhPos = (head-1 < 0) ? TOK_START_INT : sent.getPosTag(head-1);
        byte rhPos = (head+1 >= sentLen) ? TOK_END_INT : sent.getPosTag(head+1);
        //
        byte lhCpos = 0, rhCpos = 0;
        if (useCoarseTags) {
            lhCpos = (head-1 < 0) ? TOK_START_INT : sent.getCposTag(head-1);
            rhCpos = (head+1 >= sentLen) ? TOK_END_INT : sent.getCposTag(head+1);
        }
        // Two tokens to the left (ll) and right (rr).
        //
        byte llhPos = (head-2 < 0) ? TOK_START_INT : sent.getPosTag(head-2);
        byte rrhPos = (head+2 >= sentLen) ? TOK_END_INT : sent.getPosTag(head+2);
        //
        byte llhCpos = 0, rrhCpos = 0;
        if (useCoarseTags) {
            llhCpos = (head-2 < 0) ? TOK_START_INT : sent.getCposTag(head-2);
            rrhCpos = (head+2 >= sentLen) ? TOK_END_INT : sent.getCposTag(head+2);
        }
        
        // Flags for the type of feature. (NOTE: Currently, these are only used for the offset in addSimpleTokenFeatures).
        byte flags = 0;

        // --------------------------------------------------------------------
        // Bias Feature.
        // --------------------------------------------------------------------        
        addFeat(feats, mod, encodeFeatureB___(TokTs.BIAS, flags, (byte)0));

        // --------------------------------------------------------------------
        // Unigram Features.
        // --------------------------------------------------------------------
        
        addSimpleTokenFeatures(sent, head, (byte)0, feats, mod, useLemmaFeats, useCoarseTags, tagConfig);
        if (maxTokenContext >= 1) {
            addSimpleTokenFeatures(sent, head, (byte)1, feats, mod, useLemmaFeats, useCoarseTags, tagConfig);
            addSimpleTokenFeatures(sent, head, (byte)-1, feats, mod, useLemmaFeats, useCoarseTags, tagConfig);
        }
        if (maxTokenContext >= 2) {
            addSimpleTokenFeatures(sent, head, (byte)2, feats, mod, useLemmaFeats, useCoarseTags, tagConfig);
            addSimpleTokenFeatures(sent, head, (byte)-2, feats, mod, useLemmaFeats, useCoarseTags, tagConfig);
        }
        
        // --------------------------------------------------------------------
        // Sequential Bigram and Trigram Features.
        // --------------------------------------------------------------------

        if (maxTokenContext >= 1) {
            addFeat(feats, mod, encodeFeatureSBB_(TokTs.hP_lhP, flags, tagConfig, hPos, lhPos));
            addFeat(feats, mod, encodeFeatureSBB_(TokTs.hP_rhP, flags, tagConfig, hPos, rhPos));
            if (useCoarseTags) {         
                addFeat(feats, mod, encodeFeatureSBB_(TokTs.hC_lhC, flags, tagConfig, hCpos, lhCpos));
                addFeat(feats, mod, encodeFeatureSBB_(TokTs.hC_rhC, flags, tagConfig, hCpos, rhCpos));
            }
        }
        if (maxTokenContext >= 2) {
            addFeat(feats, mod, encodeFeatureSBBB(TokTs.hP_lhP_llhP, flags, tagConfig, hPos, lhPos, llhPos));
            addFeat(feats, mod, encodeFeatureSBBB(TokTs.hP_rhP_rrhP, flags, tagConfig, hPos, rhPos, rrhPos));   
            if (useCoarseTags) {         
                addFeat(feats, mod, encodeFeatureSBBB(TokTs.hC_lhC_llhC, flags, tagConfig, hCpos, lhCpos, llhCpos));
                addFeat(feats, mod, encodeFeatureSBBB(TokTs.hC_rhC_rrhC, flags, tagConfig, hCpos, rhCpos, rrhCpos));
            }
        }
    }

    /**
     * Features of a single token positioned at center+offset. The offset is included in the flags.
     */
    private static void addSimpleTokenFeatures(final IntAnnoSentence sent, final int center, final byte offset, 
            final FeatureVector feats, final int mod,
            final boolean useLemmaFeats,
            final boolean useCoarseTags,
            final short tagConfig) {
        int sentLen = sent.size();
        final int head = center + offset;
        final byte flags = offset;
        
        // Positional features.
        byte hFirst = (byte) ((head == 0) ? 1 : 0);

        // Word properties.
        byte hCap = (head < 0 || head >= sentLen) ? 0 : (byte)(sent.isCapitalized(head) ? 1 : 0); // 1-bit
        
        // words/tags/lemmas.
        short hWord = (head < 0) ? TOK_START_INT : (head >= sentLen) ? TOK_END_INT : sent.getWord(head);
        byte hPos = (head < 0) ? TOK_START_INT : (head >= sentLen) ? TOK_END_INT : sent.getPosTag(head);
        short hLemma = 0;
        if (useLemmaFeats) {
            hLemma = (head < 0) ? TOK_START_INT : (head >= sentLen) ? TOK_END_INT : sent.getLemma(head);
        }
        byte hCpos = 0;
        if (useCoarseTags) {
            hCpos = (head < 0) ? TOK_START_INT : (head >= sentLen) ? TOK_END_INT : sent.getCposTag(head);
        }
        
        byte len;
        // Prefixes
        len = 1;
        short hPre1 = (head < 0) ? TOK_START_INT : (head >= sentLen) ? TOK_END_INT : sent.getPrefix(head, len);
        // Suffixes
        len = 3;
        short hSuf3 = (head < 0) ? TOK_START_INT : (head >= sentLen) ? TOK_END_INT : sent.getSuffix(head, len);
        
        // --------------------------------------------------------------------
        // Unigram features only.
        // --------------------------------------------------------------------        
        
        // Head Only.
        addFeat(feats, mod, encodeFeatureSS__(TokTs.hW, flags, tagConfig, hWord));
        
        // Position.
        addFeat(feats, mod, encodeFeatureSB__(TokTs.hFirst, flags, tagConfig, hFirst));

        // Word properties.
        addFeat(feats, mod, encodeFeatureSB__(TokTs.hCap, flags, tagConfig, hCap));
        
        // Word.
        addFeat(feats, mod, encodeFeatureSS__(TokTs.hW, flags, tagConfig, hWord));
        
        // Prefix.
        len = 1;
        addFeat(feats, mod, encodeFeatureSSB_(TokTs.hPr, flags, tagConfig, hPre1, len));

        // Suffix.
        len = 3;
        addFeat(feats, mod, encodeFeatureSSB_(TokTs.hSu, flags, tagConfig, hSuf3, len));

        // POS Tag
        addFeat(feats, mod, encodeFeatureSB__(TokTs.hP, flags, tagConfig, hPos));
        addFeat(feats, mod, encodeFeatureSSB_(TokTs.hW_hP, flags, tagConfig, hWord, hPos));
        
        // Coarse POS Tag
        if (useCoarseTags) {
            addFeat(feats, mod, encodeFeatureSB__(TokTs.hC, flags, tagConfig, hCpos));
            addFeat(feats, mod, encodeFeatureSSB_(TokTs.hW_hC, flags, tagConfig, hWord, hCpos));            
        }
        
        // Lemma
        if (useLemmaFeats) {
            addFeat(feats, mod, encodeFeatureSS__(TokTs.hL, flags, tagConfig, hLemma));
        }
    }
    
    public static void addFeat(FeatureVector feats, int mod, long feat, double value) {
        int hash = MurmurHash.hash32(feat);
        if (mod > 0) {
            hash = FastMath.mod(hash, mod);
        }
        feats.add(hash, value);
    }
    
    public static void addFeat(FeatureVector feats, int mod, long feat) {
        int hash = MurmurHash.hash32(feat);
        if (mod > 0) {
            hash = FastMath.mod(hash, mod);
        }
        feats.add(hash, 1.0);
        // Enable this for debugging of feature creation.
        //        if (feats instanceof LongFeatureVector) {
        //            ((LongFeatureVector)feats).addLong(feat, 1.0);
        //        }
    }
    
}
