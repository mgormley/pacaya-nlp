package edu.jhu.nlp.relations;

import static org.junit.Assert.assertEquals;

import java.util.List;

import org.junit.Test;

import edu.jhu.nlp.data.NerMention;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.relations.RelationsFactorGraphBuilder.RelVar;
import edu.jhu.nlp.relations.RelWordFeatures.EmbFeatType;
import edu.jhu.nlp.relations.RelWordFeatures.RelWordFeaturesPrm;
import edu.jhu.pacaya.gm.feat.FeatureVector;
import edu.jhu.pacaya.gm.model.Var.VarType;
import edu.jhu.pacaya.util.FeatureNames;
import edu.jhu.pacaya.util.collections.QLists;
import edu.jhu.prim.map.IntDoubleEntry;
import edu.jhu.prim.tuple.Pair;


public class RelWordFeaturesTest {

    @Test
    public void testGetPerWordFeats() throws Exception {
        AnnoSentence sent = RelationMungerTest.getSentWithRelationsAndNer();
        sent.setParents(new int[]{-1, 0, 1, 2, 3, 4});
        Pair<NerMention, NerMention> pair = sent.getNePairs().get(0);
        RelVar rv = new RelVar(VarType.PREDICTED, "rv", pair.get1(), pair.get2(), 
                QLists.getList("OWNER", "NEAR", "ART"));
        
        FeatureNames alphabet = new FeatureNames();
        RelWordFeaturesPrm prm = new RelWordFeaturesPrm();
        prm.embFeatType = EmbFeatType.HEAD_TYPE_LOC;
        prm.entityTypeRepl = null;
        RelWordFeatures fe = new RelWordFeatures(prm, sent, alphabet);
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
            + "\n"
            + "FeatureVector: 1\n"
            + "-2_ne1=1.0, \n"
            + "\n"
            + "FeatureVector: 2\n"
            + "-1_ne1=1.0, \n"
            + "\n"
            + "FeatureVector: 3\n"
            + "on_path=1.0, \n"
            + "on_path-t1MAMMAL=1.0, \n"
            + "on_path-t2LOCATION=1.0, \n"
            + "on_path-t1t2MAMMALLOCATION=1.0, \n"
            + "-2_ne2=1.0, \n"
            + "ne1_head-t1MAMMAL=1.0, \n"
            + "ne1_head-t2LOCATION=1.0, \n"
            + "ne1_head-t1t2MAMMALLOCATION=1.0, \n"
            + "ne1_head=1.0, \n"
            + "\n"
            + "FeatureVector: 4\n"
            + "in_between=1.0, \n"
            + "in_between-t1MAMMAL=1.0, \n"
            + "in_between-t2LOCATION=1.0, \n"
            + "in_between-t1t2MAMMALLOCATION=1.0, \n"
            + "on_path=1.0, \n"
            + "on_path-t1MAMMAL=1.0, \n"
            + "on_path-t2LOCATION=1.0, \n"
            + "on_path-t1t2MAMMALLOCATION=1.0, \n"
            + "+1_ne1=1.0, \n"
            + "-1_ne2=1.0, \n"
            + "\n"
            + "FeatureVector: 5\n"
            + "on_path=1.0, \n"
            + "on_path-t1MAMMAL=1.0, \n"
            + "on_path-t2LOCATION=1.0, \n"
            + "on_path-t1t2MAMMALLOCATION=1.0, \n"
            + "+2_ne1=1.0, \n"
            + "ne2_head-t1MAMMAL=1.0, \n"
            + "ne2_head-t2LOCATION=1.0, \n"
            + "ne2_head-t1t2MAMMALLOCATION=1.0, \n"
            + "ne2_head=1.0, \n"
            + "\n";    
    
}
