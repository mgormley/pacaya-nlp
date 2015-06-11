package edu.jhu.nlp.depparse;

import edu.jhu.nlp.CorpusStatistics;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.data.simple.AnnoSentenceCollection;
import edu.jhu.nlp.data.simple.AnnoSentenceReaderSpeedTest;
import edu.jhu.nlp.depparse.BitshiftDepParseFeatureExtractor.BitshiftDepParseFeatureExtractorPrm;
import edu.jhu.nlp.depparse.DepParseFactorGraphBuilder.DepParseFactorGraphBuilderPrm;
import edu.jhu.nlp.depparse.DepParseFeatureExtractor.DepParseFeatureExtractorPrm;
import edu.jhu.nlp.features.TemplateSets;
import edu.jhu.pacaya.gm.data.UFgExample;
import edu.jhu.pacaya.gm.data.UnlabeledFgExample;
import edu.jhu.pacaya.gm.feat.FeatureExtractor;
import edu.jhu.pacaya.gm.feat.ObsFeatureConjoiner;
import edu.jhu.pacaya.gm.model.FactorGraph;
import edu.jhu.prim.util.Timer;

public class DepParseFactorGraphBuilderSpeedTest {
    
    /**
     * Speed test results:
     * Tokens / sec: 8028.895184135978
     */
    //@Test
    public void testSpeed() {
        AnnoSentenceCollection sents = AnnoSentenceReaderSpeedTest.readPtbYmConllx();
        
        Timer t = new Timer();
        t.start();
        for (AnnoSentence sent : sents) {
            UFgExample ex = get1stOrderFg(sent);
            ex.getFactorGraph();
        }
        t.stop();
        System.out.println("Tokens / sec: " + (sents.getNumTokens() / t.totSec()));
    }

    public static UFgExample get1stOrderFg(AnnoSentence sent) {
        // Construct a dummy feature extractor with null values.
        return get1stOrderFg(sent, null, null, 0, true);
    }
    
    public static UFgExample get1stOrderFg(AnnoSentence sent, CorpusStatistics cs, ObsFeatureConjoiner ofc, int numParams, boolean onlyFast) {
        FactorGraph fg = new FactorGraph();
        DepParseFeatureExtractorPrm fePrm = new DepParseFeatureExtractorPrm();        
        fePrm.featureHashMod = numParams;
        fePrm.firstOrderTpls = TemplateSets.getFromResource(TemplateSets.mcdonaldDepFeatsResource);
        BitshiftDepParseFeatureExtractorPrm bsFePrm = new BitshiftDepParseFeatureExtractorPrm();
        bsFePrm.featureHashMod = numParams;
        bsFePrm.useCoarseTags = true;
        FeatureExtractor fe = onlyFast?
                new BitshiftDepParseFeatureExtractor(bsFePrm, sent, cs, ofc) :
                new DepParseFeatureExtractor(fePrm, sent, cs, ofc.getFeAlphabet());
        
        DepParseFactorGraphBuilderPrm fgPrm = new DepParseFactorGraphBuilderPrm();
        fgPrm.useProjDepTreeFactor = true;
        fgPrm.grandparentFactors = false;
        fgPrm.arbitrarySiblingFactors = false;
        DepParseFactorGraphBuilder builder = new DepParseFactorGraphBuilder(fgPrm);
        builder.build(sent, fe, fg);
        
        UnlabeledFgExample ex = new UnlabeledFgExample(fg);
        return ex;
    }
    
    public static void main(String[] args) {
        (new DepParseFactorGraphBuilderSpeedTest()).testSpeed();
    }
    
}
