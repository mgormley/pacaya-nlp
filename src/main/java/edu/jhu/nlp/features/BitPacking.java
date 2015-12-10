package edu.jhu.nlp.features;

import edu.jhu.prim.Primitives;

public final class BitPacking {

    private BitPacking() { 
        // private constructor.
    }
    
    private static final long BYTE_MAX = Primitives.LONG_MAX_UBYTE;   // 0xffL;
    private static final long SHORT_MAX = Primitives.LONG_MAX_USHORT; // 0xffffL;
    private static final long INT_MAX = Primitives.LONG_MAX_UINT;     // 0xffffffffL;

    public static final long encodeFeatureS___(byte template, byte flags, short s1) {
        return (template & BYTE_MAX) | ((flags & BYTE_MAX) << 8) | ((s1 & SHORT_MAX) << 16);
    }
    
    public static final long encodeFeatureB___(byte template, byte flags, byte b1) {
        return (template & BYTE_MAX) | ((flags & BYTE_MAX) << 8) | ((b1 & BYTE_MAX) << 16);
    }
    
    public static final long encodeFeatureSB__(byte template, byte flags, short s1, byte b2) {
        return (template & BYTE_MAX) | ((flags & BYTE_MAX) << 8) | ((s1 & SHORT_MAX) << 16) | ((b2 & BYTE_MAX) << 32);
    }

    public static final long encodeFeatureSS__(byte template, byte flags, short s1, short s2) {
        return (template & BYTE_MAX) | ((flags & BYTE_MAX) << 8) | ((s1 & SHORT_MAX) << 16) | ((s2 & SHORT_MAX) << 32);
    }

    public static final long encodeFeatureBB__(byte template, byte flags, byte b1, byte b2) {
        return (template & BYTE_MAX) | ((flags & BYTE_MAX) << 8) | ((b1 & SHORT_MAX) << 16) | ((b2 & SHORT_MAX) << 24);
    }

    public static final long encodeFeatureSSB_(byte template, byte flags, short s1, short s2, byte b3) {
        return (template & BYTE_MAX) | ((flags & BYTE_MAX) << 8) | ((s1 & SHORT_MAX) << 16) | ((s2 & SHORT_MAX) << 32)
                | ((b3 & BYTE_MAX) << 48);
    }

    public static final long encodeFeatureSSS_(byte template, byte flags, short s1, short s2, short s3) {
        return (template & BYTE_MAX) | ((flags & BYTE_MAX) << 8) | ((s1 & SHORT_MAX) << 16) | ((s2 & SHORT_MAX) << 32)
                | ((s3 & SHORT_MAX) << 48);
    }
    
    public static final long encodeFeatureSBB_(byte template, byte flags, short s1, byte b2, byte b3) {
        return (template & BYTE_MAX) | ((flags & BYTE_MAX) << 8) | ((s1 & SHORT_MAX) << 16) 
                | ((b2 & BYTE_MAX) << 32) | ((b3 & BYTE_MAX) << 40);
    }
    
    public static final long encodeFeatureSBBBB(byte template, byte flags, short s1, byte b2, byte b3, byte b4, byte b5) {
        return (template & BYTE_MAX) | ((flags & BYTE_MAX) << 8) | ((s1 & SHORT_MAX) << 16) 
                | ((b2 & BYTE_MAX) << 32) | ((b3 & BYTE_MAX) << 40) | ((b4 & BYTE_MAX) << 48) 
                | ((b5 & BYTE_MAX) << 56); // Full.
    }
    
    public static final long encodeFeatureSSBB(byte template, byte flags, short s1, short s2, byte b3, byte b4) {
        return (template & BYTE_MAX) | ((flags & BYTE_MAX) << 8) | ((s1 & SHORT_MAX) << 16) | ((s2 & SHORT_MAX) << 32)
                | ((b3 & BYTE_MAX) << 48) | ((b4 & BYTE_MAX) << 56); // Full.
    }

    public static final long encodeFeatureBBB_(byte template, byte flags, byte b1, byte b2, byte b3) {
        return (template & BYTE_MAX) | ((flags & BYTE_MAX) << 8) | ((b1 & BYTE_MAX) << 16) | ((b2 & BYTE_MAX) << 24)
                | ((b3 & BYTE_MAX) << 32);
    }
    
    public static final long encodeFeatureBBBB(byte template, byte flags, byte b1, byte b2, byte b3, byte b4) {
        return (template & BYTE_MAX) | ((flags & BYTE_MAX) << 8) | ((b1 & BYTE_MAX) << 16) | ((b2 & BYTE_MAX) << 24)
                | ((b3 & BYTE_MAX) << 32) | ((b4 & BYTE_MAX) << 40);
    }
    
    public static final long encodeFeatureBBBBB(byte template, byte flags, byte b1, byte b2, byte b3, byte b4, byte b5) {
        return (template & BYTE_MAX) | ((flags & BYTE_MAX) << 8) | ((b1 & BYTE_MAX) << 16) | ((b2 & BYTE_MAX) << 24)
                | ((b3 & BYTE_MAX) << 32) | ((b4 & BYTE_MAX) << 40) | ((b5 & BYTE_MAX) << 48);
    }
    
    public static final long encodeFeatureBBBBBB(byte template, byte flags, byte b1, byte b2, byte b3, byte b4, byte b5, byte b6) {
        return (template & BYTE_MAX) | ((flags & BYTE_MAX) << 8) | ((b1 & BYTE_MAX) << 16) | ((b2 & BYTE_MAX) << 24)
                | ((b3 & BYTE_MAX) << 32) | ((b4 & BYTE_MAX) << 40) | ((b5 & BYTE_MAX) << 48) | ((b6 & BYTE_MAX) << 56); // Full.
    }

    public static final long encodeFeatureII__(int f1, int f2) {
        long feat =  (f1 & INT_MAX) | ((f2 & INT_MAX) << 32);
        return feat;
    }
    
}
