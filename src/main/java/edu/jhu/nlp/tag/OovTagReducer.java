package edu.jhu.nlp.tag;

import edu.jhu.prim.bimap.IntObjectBimap;

public class OovTagReducer extends AbstractTagReducer {
    
    private static final long serialVersionUID = 1L;
    private IntObjectBimap<String> alphabet;
    private String unk;
    
    public OovTagReducer(IntObjectBimap<String> alphabet, String unk) {
        super();
        this.alphabet = alphabet;
        this.unk = unk;
    }

    @Override
    public String reduceTag(String tag) {
        if (alphabet.lookupIndex(tag) == -1) {
            return unk;
        }
        return tag;
    }

}
