package edu.jhu.nlp.sprl;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedList;

import org.junit.Test;

import edu.jhu.nlp.CorpusStatistics;
import edu.jhu.nlp.CorpusStatistics.CorpusStatisticsPrm;
import edu.jhu.nlp.ObsFeTypedFactor;
import edu.jhu.nlp.data.concrete.ConcreteReader;
import edu.jhu.nlp.data.concrete.ConcreteReader.ConcreteReaderPrm;
import edu.jhu.nlp.data.simple.AlphabetStore;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.data.simple.AnnoSentenceCollection;
import edu.jhu.nlp.data.simple.IntAnnoSentence;
import edu.jhu.nlp.features.TemplateLanguage.AT;
import edu.jhu.nlp.features.TemplateLanguage.FeatTemplate1;
import edu.jhu.nlp.features.TemplateLanguage.Position;
import edu.jhu.nlp.features.TemplateLanguage.PositionModifier;
import edu.jhu.nlp.features.TemplateLanguage.TokProperty;
import edu.jhu.nlp.sprl.SprlFactorGraphBuilder.SprlFactorGraphBuilderPrm;
import edu.jhu.nlp.sprl.SprlFactorGraphBuilder.SprlFactorType;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.RoleStructure;
import edu.jhu.nlp.srl.SrlFeatureExtractor.SrlFeatureExtractorPrm;
import edu.jhu.pacaya.gm.data.FgExampleList;
import edu.jhu.pacaya.gm.data.FgExampleListBuilder;
import edu.jhu.pacaya.gm.data.FgExampleListBuilder.FgExamplesBuilderPrm;
import edu.jhu.pacaya.gm.data.LFgExample;
import edu.jhu.pacaya.gm.data.LabeledFgExample;
import edu.jhu.pacaya.gm.feat.FactorTemplateList;
import edu.jhu.pacaya.gm.feat.FeatureVector;
import edu.jhu.pacaya.gm.feat.ObsFeatureConjoiner;
import edu.jhu.pacaya.gm.feat.ObsFeatureConjoiner.ObsFeatureConjoinerPrm;
import edu.jhu.pacaya.gm.model.FactorGraph;
import edu.jhu.pacaya.gm.model.VarConfig;
import edu.jhu.prim.tuple.Triple;

public class SprlFactorGraphBuilderTest {
    private static String concreteFilename = "/edu/jhu/nlp/data/concrete/minisprl.comm";

    @Test
    public void testCounts() throws IOException {
        File f = new File(getClass().getResource(concreteFilename).getFile());
        ConcreteReaderPrm crPrm = new ConcreteReaderPrm();
        crPrm.depParseTool = null;
        SprlLabelConverter sprlConverter = new BinarySprlLabelConverter(1.5); 
        crPrm.sprlConverter = sprlConverter; 
        crPrm.srlTool = "fpropbank";
        crPrm.sprlTool = "fpropbank";
        ConcreteReader r = new ConcreteReader(crPrm);
        AnnoSentenceCollection sents = r.sentsFromCommFile(f);
        assertEquals(2, sents.size());
        SprlFactorGraphBuilderPrm prm = new SprlFactorGraphBuilderPrm();
        prm.allowPredArgSelfLoops = true;
        prm.labelConverter = sprlConverter; 
        prm.pairwiseFactors = false;
        prm.roleStructure = RoleStructure.PAIRS_GIVEN;
        SrlFeatureExtractorPrm srlFePrm = new SrlFeatureExtractorPrm();
        srlFePrm.argTemplates = Arrays.asList(new FeatTemplate1(Position.PARENT, PositionModifier.HEAD, TokProperty.CAPITALIZED));
        srlFePrm.useTemplates = false;
//        srlFePrm.featureHashMod = 1000003;
        srlFePrm.featureHashMod = -1;
        srlFePrm.senseTemplates = Arrays.asList(new FeatTemplate1(Position.PARENT, PositionModifier.HEAD, TokProperty.LEMMA));
        srlFePrm.biasOnly = false;
        prm.srlFePrm = srlFePrm;
        prm.unaryFactors = true;
        SprlFactorGraphBuilder builder = new SprlFactorGraphBuilder(prm);
        ObsFeatureConjoinerPrm ofcPrm = new ObsFeatureConjoinerPrm();
        ofcPrm.featCountCutoff = 1;
        ofcPrm.includeUnsupportedFeatures = false;
        //FgExampleMemoryStore data = new FgExampleMemoryStore();
        //data.add(new LabeledFgExample(fg, trainConfig, fts));
        //ofc.init(data);

        FgExamplesBuilderPrm exListPrm = new FgExamplesBuilderPrm();
        FgExampleListBuilder exListBuilder = new FgExampleListBuilder(exListPrm);

        CorpusStatisticsPrm csPrm = new CorpusStatisticsPrm();
        csPrm.cutoff = 0;
        csPrm.language = "en";
        csPrm.normalizeWords = false;
        csPrm.topN = 100;
        csPrm.useGoldSyntax = true;
        CorpusStatistics cs = new CorpusStatistics(csPrm);
        
        ObsFeatureConjoiner ofc = new ObsFeatureConjoiner(ofcPrm, new FactorTemplateList());
        if (!cs.isInitialized()) {
            cs.init(sents, false);
            cs.store = new AlphabetStore(sents);
        }
        FgExampleList exampleList = exListBuilder.getInstance(new FgExampleList() {

            @Override
            public int size() {
                return sents.size();
            }
            
            @Override
            public LFgExample get(int i) {
                IntAnnoSentence isent = new IntAnnoSentence(sents.get(i), cs.store);
                FactorGraph fg = new FactorGraph(); 
                builder.build(isent, ofc, fg, cs);
                VarConfig vc = new VarConfig();
                builder.annoToConfig(sents.get(i), vc);
                AnnoSentence copy = sents.get(i).getShallowCopy();
                copy.removeAt(AT.SPRL);
                builder.configToAnno(vc, copy);
                SprlProperties gold = sents.get(0).getSprl();
                for (Triple<Integer, Integer, String> t : gold.getLabeledProperties()) {
                    assertEquals(gold.get(t), copy.getSprl().get(t));
                }
                return new LabeledFgExample(fg, vc, ofc.getTemplates());
            }
        });
        ofc.init(exampleList);
        IntAnnoSentence isent = new IntAnnoSentence(sents.get(0), cs.store);
        FactorGraph fg = new FactorGraph(); 
        builder.build(isent, ofc, fg, cs);
        assertEquals(4, builder.getSprlVars().length);
        assertEquals(4, builder.getSprlVars()[0].length);
        assertEquals(2, builder.getSprlVars()[0][0].length);
        assertEquals(2, builder.getSprlVars()[0][1].length);
        assertTrue(builder.getSprlVars()[0][0][0] == null);
        assertTrue(builder.getSprlVars()[0][0][1] == null);
        assertTrue(builder.getSprlVars()[0][1][0] == null);
        assertTrue(builder.getSprlVars()[0][1][1] == null);
        assertTrue(builder.getSprlVars()[1][0][0] != null);
        assertTrue(builder.getSprlVars()[1][0][1] != null);
        assertTrue(builder.getSprlVars()[1][2][0] != null);
        assertTrue(builder.getSprlVars()[1][2][1] != null);
        // 2 pairs each with 2 properties; so, 2 unary factors
        assertEquals(4, fg.getNumVars());
        assertEquals(4, fg.getNumFactors());
        assertEquals(4, fg.getFactors().size());
        //fg.getFactors().get(0).g
        ObsFeTypedFactor factor = ((ObsFeTypedFactor)fg.getFactor(0));
        factor.fill(1);
        LinkedList<Serializable> templateKey = new LinkedList<>();
        Collections.addAll(templateKey, SprlFactorType.SPRL_UNARY, "awareness");
        assertEquals(templateKey, factor.getTemplateKey());
        FeatureVector fv = builder.getFeatExtractor().calcObsFeatureVector(factor);
        assertEquals(2, fv.getUsed());
        assertEquals(2, fv.getNumImplicitEntries());
        assertEquals(4, ofc.getTemplates().getNumObsFeats());
        assertEquals(templateKey, ofc.getTemplates().get(0).getKey());
        assertEquals(2, ofc.getTemplates().get(0).getNumConfigs());
        assertEquals(8, ofc.getNumParams());
        
    }

}
