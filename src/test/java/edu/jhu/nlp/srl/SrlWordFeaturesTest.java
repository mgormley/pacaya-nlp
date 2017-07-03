package edu.jhu.nlp.srl;

import static org.junit.Assert.assertEquals;

import java.util.List;

import org.junit.Test;

import edu.jhu.nlp.data.conll.CoNLL09Sentence;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.data.simple.AnnoSentenceTest;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.RoleVar;
import edu.jhu.nlp.srl.SrlWordFeatures.SrlWordFeatType;
import edu.jhu.nlp.srl.SrlWordFeatures.SrlWordFeaturesPrm;
import edu.jhu.pacaya.gm.feat.FeatureVector;
import edu.jhu.pacaya.gm.model.Var.VarType;
import edu.jhu.pacaya.util.FeatureNames;
import edu.jhu.pacaya.util.collections.QLists;
import edu.jhu.prim.map.IntDoubleEntry;


public class SrlWordFeaturesTest {

    @Test
    public void testGetPerWordFeats() throws Exception {
        CoNLL09Sentence csent = AnnoSentenceTest.getDogConll09Sentence();
        AnnoSentence sent = csent.toAnnoSentence(true);
        RoleVar rv = new RoleVar(VarType.PREDICTED, 3, "rv", QLists.getList("arg0", "arg1", "arg2"), 2, 0);
        sent.setClusters(QLists.getList("010101", "01", "101010", "010101"));
        
        FeatureNames alphabet = new FeatureNames();
        SrlWordFeaturesPrm prm = new SrlWordFeaturesPrm();
        prm.srlEmbFeatType = SrlWordFeatType.HEAD_TYPE_LOC;
        SrlWordFeatures fe = new SrlWordFeatures(prm, sent, alphabet);
        List<FeatureVector> featsList = fe.getFeatures(rv);
        
        assertEquals(sent.size(), featsList.size());
        String fs = "";
        for (int i=0; i<featsList.size(); i++) {
            //for (int k=0; k<featsListSystem.out.println()
            FeatureVector feats = featsList.get(i);
            fs += "FeatureVector: " + i + "\n";
            for (IntDoubleEntry e : feats) {
                fs += alphabet.lookupObject(e.index()) + "=" + e.get() + ", \n";
            }
            fs += "\n";
        }
        System.out.println(fs);
        assertEquals(expectedFs, fs);
    }
    String expectedFs = "FeatureVector: 0\n"
            + "on_path=1.0, \n"
//            + "on_path-hC:1010=1.0, \n"
//            + "on_path-mC:0101=1.0, \n"
//            + "on_path-hCmC:10100101=1.0, \n"
            + "-2_h=1.0, \n"
//            + "l-hC:1010=1.0, \n"
//            + "l-mC:0101=1.0, \n"
//            + "l-hCmC:10100101=1.0, \n"
            + "m-hC:1010=1.0, \n"
            + "m-mC:0101=1.0, \n"
            + "m-hCmC:10100101=1.0, \n"
            + "m=1.0, \n"
            + "\n"
            + "FeatureVector: 1\n"
            + "in_between=1.0, \n"
//            + "in_between-hC:1010=1.0, \n"
//            + "in_between-mC:0101=1.0, \n"
//            + "in_between-hCmC:10100101=1.0, \n"
            + "on_path=1.0, \n"
//            + "on_path-hC:1010=1.0, \n"
//            + "on_path-mC:0101=1.0, \n"
//            + "on_path-hCmC:10100101=1.0, \n"
            + "-1_h=1.0, \n"
            + "+1_m=1.0, \n"
            + "\n"
            + "FeatureVector: 2\n"
            + "on_path=1.0, \n"
//            + "on_path-hC:1010=1.0, \n"
//            + "on_path-mC:0101=1.0, \n"
//            + "on_path-hCmC:10100101=1.0, \n"
            + "+2_m=1.0, \n"
//            + "r-hC:1010=1.0, \n"
//            + "r-mC:0101=1.0, \n"
//            + "r-hCmC:10100101=1.0, \n"
            + "h-hC:1010=1.0, \n"
            + "h-mC:0101=1.0, \n"
            + "h-hCmC:10100101=1.0, \n"
            + "h=1.0, \n"
            + "\n"
            + "FeatureVector: 3\n"
            + "+1_h=1.0, \n"
            + "\n";
    
}
