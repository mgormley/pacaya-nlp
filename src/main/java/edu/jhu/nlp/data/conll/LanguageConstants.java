package edu.jhu.nlp.data.conll;

/**
 * CoNLL-2009 Shared Task dataset: Language and corpus specific constants.
 * 
 * @author mgormley
 */
public class LanguageConstants {

    public static boolean isNoun(String pos, String language) {
        if ("es".equals(language) || "ca".equals(language)) {
            // The full Ancora Spanish/Catalan tagset is described in "tagset.pdf" which is
            // included with the CoNLL-2009 Trial Data download here:
            // http://ufal.mff.cuni.cz/conll2009-st/trial-data.html
            return pos.equals("n");
        } else if ("de".equals(language)) {
            // The full TIGER German tagset is described in
            // "A brief introduction to the TIGER Treebank, version 1" (Smith,
            // 2003).
            // http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.232.9998
            return pos.startsWith("N");
        } else if ("cs".equals(language)) {
            // The POS tag is the first position of a 15 character tag. Page 15
            // describes each POS tag used by CoNLL in
            // "A Manual for Morphological Annotation, 2nd edition (html)"
            // (Zeman et al., 2005).
            // http://ufal.mff.cuni.cz/pdt2.0/doc/manuals/en/m-layer/pdf/m-man-en.pdf
            return pos.equals("N");
        } else if ("en".equals(language)) {
            // The Penn Treebank tagset is described here:
            // http://bulba.sdsu.edu/jeanette/thesis/PennTags.html
            // ftp://ftp.cis.upenn.edu/pub/treebank/doc/tagguide.ps.gz
            return pos.startsWith("N");
        } else if ("zh".equals(language)) {
            // The Penn Chinese Treebank tagset is described here:
            // http://verbs.colorado.edu/chinese/posguide.3rd.ch.pdf
            return pos.startsWith("N");
        } else {
            throw new RuntimeException("Unknown language: " + language);
        }
    }
    
    public static boolean isVerb(String pos, String language) {
        if ("es".equals(language) || "ca".equals(language)) {
            return pos.equals("v");
        } else if ("de".equals(language)) {
            return pos.startsWith("V");
        } else if ("cs".equals(language)) {
            return pos.equals("V");
        } else if ("en".equals(language)) {
            return pos.startsWith("V");
        } else if ("zh".equals(language)) {
            return pos.startsWith("V");
        } else {
            throw new RuntimeException("Unknown language: " + language);
        }
    }
    
}
