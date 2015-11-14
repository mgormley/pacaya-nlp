package edu.jhu.nlp.relations;

import java.util.Collection;
import java.util.List;

import edu.jhu.pacaya.gm.feat.FeatureVector;
import edu.jhu.pacaya.util.FeatureNames;
import edu.jhu.pacaya.util.hash.MurmurHash3;
import edu.jhu.prim.list.IntArrayList;
import edu.jhu.prim.util.math.FastMath;

public class FeatureUtils {

    /**
     * Adds string features to a feature vector with one of two options:
     * <ul>
     * <li>Option 1: Remaps the strings to ints then adds them.</li>
     * <li>Option 2: Hashes, mods, remaps, then adds the String features to a feature vector. This
     * option will additionally compute the hash of the reversed string and add either +1 or -1
     * following (Weinberger et al., 2009).</li>
     * </ul>
     */
    public static void addFeatures(Collection<String> obsFeats, FeatureNames alphabet, FeatureVector fv, boolean isBiasFeat, int featureHashMod) {
        if (featureHashMod <= 0) {
            // Just use the features as-is.
            for (String fname : obsFeats) {
                int fidx = alphabet.lookupIndex(fname);
                if (isBiasFeat) { alphabet.setIsBias(fidx); }
                if (fidx != -1) {
                    fv.add(fidx, 1.0);
                }
            }
        } else {
            // Apply the feature-hashing trick and store the ints in the Alphabet.
            for (String fname : obsFeats) {
                int hash = MurmurHash3.murmurhash3_x86_32(fname);
                hash = FastMath.mod(hash, featureHashMod);
                int fidx = alphabet.lookupIndex(hash);
                if (isBiasFeat) { alphabet.setIsBias(fidx); }
                if (fidx != -1) {
                    int revHash = FeatureUtils.reverseHashCode(fname);
                    if (revHash < 0) {
                        fv.add(fidx, -1.0);
                    } else {
                        fv.add(fidx, 1.0);
                    }
                }
            }
        }
    }

    /** Hashes, mods, then adds the String features to a feature vector. */
    public static void addFeatures(Collection<String> feats, FeatureVector fv, int featureHashMod) {
        // Apply the feature-hashing trick and ignore the alphabet.
        assert featureHashMod > 0;
        for (String fname : feats) {
            int hash = MurmurHash3.murmurhash3_x86_32(fname);
            int fidx = FastMath.mod(hash, featureHashMod);
            fv.add(fidx, 1.0);
            // ALERT: no reverse hashing here.
        }
    }

    /** Mods then adds the int features to a feature vector. */
    public static void addFeatures(IntArrayList feats, FeatureVector fv, int featureHashMod) {
        // Apply the feature-hashing trick and ignore the alphabet.
        assert featureHashMod > 0;
        for (int k=0; k<feats.size(); k++) {
            int hash = feats.get(k);
            int fidx = FastMath.mod(hash, featureHashMod);
            fv.add(fidx, 1.0);
            // ALERT: no reverse hashing here.
        }
    }
    
    /** Mods, re-maps, then adds int features to a feature vector. */
    public static void addFeatures(IntArrayList feats, FeatureVector fv, int featureHashMod, FeatureNames alphabet) {
        // Apply the feature-hashing trick and use the alphabet.
        assert featureHashMod > 0;
        for (int k=0; k<feats.size(); k++) {
            int hash = feats.get(k);
            hash = FastMath.mod(hash, featureHashMod);
            int fidx = alphabet.lookupIndex(hash);
            if (fidx != -1) { fv.add(fidx, 1.0); }
            // ALERT: no reverse hashing here.
        }
    }

    /**
     * Returns the hash code of the reverse of this string.
     */
    public static int reverseHashCode(String fname) {
        int hash = 0;
        int n = fname.length();
        for (int i=n-1; i>=0; i--) {
            hash += 31 * hash + fname.charAt(i);
        }
        return hash;
    }
    
    /** Prepends a prefix to each string. */
    public static void addPrefix(List<String> strs, String prefix) {
        for (int i=0; i<strs.size(); i++) {
            strs.set(i, prefix + strs.get(i));
        }
    }

}
