package edu.jhu.nlp.tag;

import static edu.jhu.nlp.data.simple.AlphabetStore.TOK_END_INT;
import static edu.jhu.nlp.data.simple.AlphabetStore.TOK_START_INT;
import static edu.jhu.nlp.data.simple.AlphabetStore.TOK_WALL_INT;
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
        public static final byte hP = next();
        public static final byte lhP = next();
        public static final byte rhP = next();
        public static final byte llhP = next();
        public static final byte rrhP = next();
        public static final byte hS = next();
        public static final byte lhS = next();
        public static final byte rhS = next();
        public static final byte llhS = next();
        public static final byte rrhS = next();
    }
    
    public static void addUnigramFeatures(IntAnnoSentence sent, int head, FeatureVector feats, int mod, short tagConfig) {
        addWordFeatures(sent, head, feats, mod, FeatureCollection.UNIGRAM, tagConfig);
    }
    
    public static void addBigramFeatures(IntAnnoSentence sent, int head, FeatureVector feats, int mod, short tagConfig) {
        byte flags = FeatureCollection.BIGRAM; // 4 bits.
        addFeat(feats, mod, encodeFeatureS___(TokTs.BIAS, flags, tagConfig));
    }
    
    private static void addWordFeatures(IntAnnoSentence sent, int head, FeatureVector feats, int mod, byte featCol, short tagConfig) {
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
        byte hCap = (byte)(sent.isCapitalized(head) ? 1 : 0); // 1-bit
        
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
        addFeat(feats, mod, encodeFeatureSSB_(TokTs.hP, flags, tagConfig, hPre1, len));
        addFeat(feats, mod, encodeFeatureSSB_(TokTs.lhP, flags, tagConfig, lhPre1, len)); 
        addFeat(feats, mod, encodeFeatureSSB_(TokTs.rhP, flags, tagConfig, rhPre1, len));
        addFeat(feats, mod, encodeFeatureSSB_(TokTs.llhP, flags, tagConfig, llhPre1, len));
        addFeat(feats, mod, encodeFeatureSSB_(TokTs.rrhP, flags, tagConfig, rrhPre1, len));

        // Suffix.
        len = 3;
        addFeat(feats, mod, encodeFeatureSSB_(TokTs.hS, flags, tagConfig, hSuf3, len));
        addFeat(feats, mod, encodeFeatureSSB_(TokTs.lhS, flags, tagConfig, lhSuf3, len)); 
        addFeat(feats, mod, encodeFeatureSSB_(TokTs.rhS, flags, tagConfig, rhSuf3, len));
        addFeat(feats, mod, encodeFeatureSSB_(TokTs.llhS, flags, tagConfig, llhSuf3, len));
        addFeat(feats, mod, encodeFeatureSSB_(TokTs.rrhS, flags, tagConfig, rrhSuf3, len));
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

    private static final long BYTE_MAX =  0xff;
    private static final long SHORT_MAX = 0xffff;
    private static final long INT_MAX =   0xffffffff;

    private static long encodeFeatureS___(byte template, byte flags, short s1) {
        return (template & BYTE_MAX) | ((flags & BYTE_MAX) << 8) | ((s1 & SHORT_MAX) << 16);
    }
    
    private static long encodeFeatureB___(byte template, byte flags, byte b1) {
        return (template & BYTE_MAX) | ((flags & BYTE_MAX) << 8) | ((b1 & BYTE_MAX) << 16);
    }
    
    private static long encodeFeatureSB__(byte template, byte flags, short s1, byte b2) {
        return (template & BYTE_MAX) | ((flags & BYTE_MAX) << 8) | ((s1 & SHORT_MAX) << 16) | ((b2 & BYTE_MAX) << 32);
    }

    private static long encodeFeatureSS__(byte template, byte flags, short s1, short s2) {
        return (template & BYTE_MAX) | ((flags & BYTE_MAX) << 8) | ((s1 & SHORT_MAX) << 16) | ((s2 & SHORT_MAX) << 32);
    }

    private static long encodeFeatureBB__(byte template, byte flags, byte b1, byte b2) {
        return (template & BYTE_MAX) | ((flags & BYTE_MAX) << 8) | ((b1 & SHORT_MAX) << 16) | ((b2 & SHORT_MAX) << 24);
    }

    private static long encodeFeatureSSB_(byte template, byte flags, short s1, short s2, byte b3) {
        return (template & BYTE_MAX) | ((flags & BYTE_MAX) << 8) | ((s1 & SHORT_MAX) << 16) | ((s2 & SHORT_MAX) << 32)
                | ((b3 & BYTE_MAX) << 48);
    }

    private static long encodeFeatureSSS_(byte template, byte flags, short s1, short s2, short s3) {
        return (template & BYTE_MAX) | ((flags & BYTE_MAX) << 8) | ((s1 & SHORT_MAX) << 16) | ((s2 & SHORT_MAX) << 32)
                | ((s3 & SHORT_MAX) << 48);
    }
    
    private static long encodeFeatureSBB_(byte template, byte flags, short s1, byte b2, byte b3) {
        return (template & BYTE_MAX) | ((flags & BYTE_MAX) << 8) | ((s1 & SHORT_MAX) << 16) 
                | ((b2 & BYTE_MAX) << 32) | ((b3 & BYTE_MAX) << 40);
    }
    
    private static long encodeFeatureSBBBB(byte template, byte flags, short s1, byte b2, byte b3, byte b4, byte b5) {
        return (template & BYTE_MAX) | ((flags & BYTE_MAX) << 8) | ((s1 & SHORT_MAX) << 16) 
                | ((b2 & BYTE_MAX) << 32) | ((b3 & BYTE_MAX) << 40) | ((b4 & BYTE_MAX) << 48) 
                | ((b5 & BYTE_MAX) << 56); // Full.
    }
    
    private static long encodeFeatureSSBB(byte template, byte flags, short s1, short s2, byte b3, byte b4) {
        return (template & BYTE_MAX) | ((flags & BYTE_MAX) << 8) | ((s1 & SHORT_MAX) << 16) | ((s2 & SHORT_MAX) << 32)
                | ((b3 & BYTE_MAX) << 48) | ((b4 & BYTE_MAX) << 56); // Full.
    }

    private static long encodeFeatureBBB_(byte template, byte flags, byte b1, byte b2, byte b3) {
        return (template & BYTE_MAX) | ((flags & BYTE_MAX) << 8) | ((b1 & BYTE_MAX) << 16) | ((b2 & BYTE_MAX) << 24)
                | ((b3 & BYTE_MAX) << 32);
    }
    
    private static long encodeFeatureBBBB(byte template, byte flags, byte b1, byte b2, byte b3, byte b4) {
        return (template & BYTE_MAX) | ((flags & BYTE_MAX) << 8) | ((b1 & BYTE_MAX) << 16) | ((b2 & BYTE_MAX) << 24)
                | ((b3 & BYTE_MAX) << 32) | ((b4 & BYTE_MAX) << 40);
    }
    
    private static long encodeFeatureBBBBB(byte template, byte flags, byte b1, byte b2, byte b3, byte b4, byte b5) {
        return (template & BYTE_MAX) | ((flags & BYTE_MAX) << 8) | ((b1 & BYTE_MAX) << 16) | ((b2 & BYTE_MAX) << 24)
                | ((b3 & BYTE_MAX) << 32) | ((b4 & BYTE_MAX) << 40) | ((b5 & BYTE_MAX) << 48);
    }
    
    private static long encodeFeatureBBBBBB(byte template, byte flags, byte b1, byte b2, byte b3, byte b4, byte b5, byte b6) {
        return (template & BYTE_MAX) | ((flags & BYTE_MAX) << 8) | ((b1 & BYTE_MAX) << 16) | ((b2 & BYTE_MAX) << 24)
                | ((b3 & BYTE_MAX) << 32) | ((b4 & BYTE_MAX) << 40) | ((b5 & BYTE_MAX) << 48) | ((b6 & BYTE_MAX) << 56); // Full.
    }
    
}
