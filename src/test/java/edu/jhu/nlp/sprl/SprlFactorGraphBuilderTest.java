package edu.jhu.nlp.sprl;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.Arrays;
import java.util.List;

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
import edu.jhu.nlp.joint.JointNlpFactorGraph.JointFactorTemplate;
import edu.jhu.nlp.sprl.SprlFactorGraphBuilder.SprlFactorGraphBuilderPrm;
import edu.jhu.nlp.sprl.SprlFactorGraphBuilder.SprlFactorType;
import edu.jhu.nlp.srl.SrlEncoder;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.RoleStructure;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.SrlFactorGraphBuilderPrm;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.SrlFactorTemplate;
import edu.jhu.nlp.srl.SrlFeatureExtractor.SrlFeatureExtractorPrm;
import edu.jhu.pacaya.gm.data.FgExampleList;
import edu.jhu.pacaya.gm.data.FgExampleListBuilder;
import edu.jhu.pacaya.gm.data.FgExampleListBuilder.FgExamplesBuilderPrm;
import edu.jhu.pacaya.gm.data.LFgExample;
import edu.jhu.pacaya.gm.data.LabeledFgExample;
import edu.jhu.pacaya.gm.feat.FactorTemplate;
import edu.jhu.pacaya.gm.feat.FactorTemplateList;
import edu.jhu.pacaya.gm.feat.FeatureVector;
import edu.jhu.pacaya.gm.feat.ObsFeatureConjoiner;
import edu.jhu.pacaya.gm.feat.ObsFeatureConjoiner.ObsFeatureConjoinerPrm;
import edu.jhu.pacaya.gm.model.FactorGraph;
import edu.jhu.pacaya.gm.model.VarConfig;
import edu.jhu.pacaya.sch.util.TestUtils;
import edu.jhu.prim.tuple.Triple;

public class SprlFactorGraphBuilderTest {
    private static String concreteFilename = "/edu/jhu/nlp/data/concrete/minisprl.comm";

    @Test
    public void test() throws IOException {
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
        
        ObsFeatureConjoinerPrm ofcPrm = new ObsFeatureConjoinerPrm();
        ofcPrm.featCountCutoff = 1;
        ofcPrm.includeUnsupportedFeatures = false;
        //FgExampleMemoryStore data = new FgExampleMemoryStore();
        //data.add(new LabeledFgExample(fg, trainConfig, fts));
        //ofc.init(data);

        FgExamplesBuilderPrm exListPrm = new FgExamplesBuilderPrm();

        CorpusStatisticsPrm csPrm = new CorpusStatisticsPrm();
        csPrm.cutoff = 0;
        csPrm.language = "en";
        csPrm.normalizeWords = false;
        csPrm.topN = 100;
        csPrm.useGoldSyntax = true;
        

        SprlFactorGraphBuilderPrm prm = new SprlFactorGraphBuilderPrm();
        prm.allowPredArgSelfLoops = true;
        prm.labelConverter = sprlConverter; 
        prm.roleStructure = RoleStructure.PAIRS_GIVEN;
        SrlFeatureExtractorPrm srlFePrm = new SrlFeatureExtractorPrm();
        srlFePrm.useTemplates = false;
//        srlFePrm.featureHashMod = 1000003;
        srlFePrm.featureHashMod = -1;
        srlFePrm.senseTemplates = Arrays.asList(new FeatTemplate1(Position.PARENT, PositionModifier.HEAD, TokProperty.LEMMA));
        srlFePrm.biasOnly = false;
        prm.unaryFactors = true;

        // only unary factors
        {
            srlFePrm.argTemplates = Arrays.asList(new FeatTemplate1(Position.PARENT, PositionModifier.HEAD, TokProperty.CAPITALIZED));
            prm.srlFePrm = srlFePrm;
            prm.pairwiseFactors = false;
            CorpusStatistics cs = new CorpusStatistics(csPrm);
            ObsFeatureConjoiner ofc = new ObsFeatureConjoiner(ofcPrm, new FactorTemplateList());
            if (!cs.isInitialized()) {
                cs.init(sents, false);
                cs.store = new AlphabetStore(sents);
            }
            FgExampleListBuilder exListBuilder = new FgExampleListBuilder(exListPrm);
            FgExampleList exampleList = exListBuilder.getInstance(new FgExampleList() {
                
                @Override
                public int size() {
                    return sents.size();
                }
                
                @Override
                public LFgExample get(int i) {
                    AnnoSentence sent = sents.get(i);
                    IntAnnoSentence isent = new IntAnnoSentence(sent, cs.store);
                    FactorGraph fg = new FactorGraph(); 
                    SprlFactorGraphBuilder builder = new SprlFactorGraphBuilder(prm);
                    builder.build(isent, ofc, fg, cs);
                    VarConfig vc = new VarConfig();
                    builder.annoToConfig(sent, vc);
                    AnnoSentence copy = sent.getShallowCopy();
                    copy.removeAt(AT.SPRL);
                    builder.configToAnno(vc, copy);
                    SprlProperties gold = sent.getSprl();
                    for (Triple<Integer, Integer, String> t : gold.getLabeledProperties()) {
                        assertEquals(gold.get(t), copy.getSprl().get(t));
                    }
                    return new LabeledFgExample(fg, vc, ofc.getTemplates());
                }
            });
            ofc.init(exampleList);
            IntAnnoSentence isent = new IntAnnoSentence(sents.get(0), cs.store);
            FactorGraph fg = new FactorGraph(); 
            SprlFactorGraphBuilder builder = new SprlFactorGraphBuilder(prm);
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
            List<Serializable> templateKey = Arrays.asList(SprlFactorType.SPRL_UNARY, "awareness");
            assertEquals(templateKey, factor.getTemplateKey());
            FeatureVector fv = builder.getFeatExtractor().calcObsFeatureVector(factor);
            assertEquals(2, fv.getUsed());
            assertEquals(2, fv.getNumImplicitEntries());
            assertEquals(4, ofc.getTemplates().getNumObsFeats());
            assertEquals(2, ofc.getTemplates().size());
            assertEquals(Arrays.asList(SprlFactorType.SPRL_UNARY, "awareness"), ofc.getTemplates().get(0).getKey());
            assertEquals(Arrays.asList(SprlFactorType.SPRL_UNARY, "volitional"), ofc.getTemplates().get(1).getKey());
            assertEquals(2, ofc.getTemplates().get(0).getNumConfigs());
            assertEquals(8, ofc.getNumParams());
        }
            
        
        // sprl pairwise
        {
            srlFePrm.argTemplates = Arrays.asList(new FeatTemplate1(Position.PARENT, PositionModifier.HEAD, TokProperty.CAPITALIZED),
                    new FeatTemplate1(Position.CHILD, PositionModifier.HEAD, TokProperty.CAPITALIZED),
                    new FeatTemplate1(Position.PARENT, PositionModifier.HEAD, TokProperty.LC),
                    new FeatTemplate1(Position.CHILD, PositionModifier.HEAD, TokProperty.LC));
                    
            prm.srlFePrm = srlFePrm;
            prm.pairwiseFactors = true;
            CorpusStatistics cs = new CorpusStatistics(csPrm);
            ObsFeatureConjoiner ofc = new ObsFeatureConjoiner(ofcPrm, new FactorTemplateList());
            if (!cs.isInitialized()) {
                cs.init(sents, false);
                cs.store = new AlphabetStore(sents);
            }
            FgExampleListBuilder exListBuilder = new FgExampleListBuilder(exListPrm);
            FgExampleList exampleList = exListBuilder.getInstance(new FgExampleList() {
                
                @Override
                public int size() {
                    return sents.size();
                }
                
                @Override
                public LFgExample get(int i) {
                    AnnoSentence sent = sents.get(i);
                    IntAnnoSentence isent = new IntAnnoSentence(sent, cs.store);
                    FactorGraph fg = new FactorGraph(); 
                    SprlFactorGraphBuilder builder = new SprlFactorGraphBuilder(prm);
                    builder.build(isent, ofc, fg, cs);
                    VarConfig vc = new VarConfig();
                    builder.annoToConfig(sent, vc);
                    AnnoSentence copy = sent.getShallowCopy();
                    copy.removeAt(AT.SPRL);
                    builder.configToAnno(vc, copy);
                    SprlProperties gold = sent.getSprl();
                    for (Triple<Integer, Integer, String> t : gold.getLabeledProperties()) {
                        assertEquals(gold.get(t), copy.getSprl().get(t));
                    }
                    return new LabeledFgExample(fg, vc, ofc.getTemplates());
                }
            });
            ofc.init(exampleList);
            IntAnnoSentence isent = new IntAnnoSentence(sents.get(0), cs.store);
            FactorGraph fg = new FactorGraph(); 
            SprlFactorGraphBuilder builder = new SprlFactorGraphBuilder(prm);
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
            // 2*2 + 2 = 6; 2 pairs each with 2 properties each with 1 unary plus 1 pairwise for each pair
            assertEquals(6, fg.getNumFactors());
            assertEquals(6, fg.getFactors().size());
            //fg.getFactors().get(0).g
            // 3 are listed below: aw, aw+vol, vol
            assertEquals(3, ofc.getTemplates().size());
            // has to do with the number of times we saw combos of the obs features
            assertEquals(13, ofc.getTemplates().getNumObsFeats());
            // 20 unary templates but 2 unseen and 4 pairwise bias templates
            assertEquals(24 - 2 + 4, ofc.getNumParams());
            assertEquals(Arrays.asList(SprlFactorType.SPRL_UNARY, "awareness"), ofc.getTemplates().get(0).getKey());
            assertEquals(Arrays.asList(SprlFactorType.SPRL_UNARY, "volitional"), ofc.getTemplates().get(1).getKey());
            assertEquals(Arrays.asList(SprlFactorType.SPRL_PAIRWISE, "volitional", "awareness"), ofc.getTemplates().get(2).getKey());
            {
                FactorTemplate template = ofc.getTemplates().get(0);                
                ObsFeTypedFactor factor = ((ObsFeTypedFactor)fg.getFactor(0));
                assertEquals(Arrays.asList(SprlFactorType.SPRL_UNARY, "awareness"), factor.getTemplateKey());
                assertEquals(2, template.getNumConfigs());
                FeatureVector fv = builder.getFeatExtractor().calcObsFeatureVector(factor);
                // 4 + bias
                assertEquals(5, fv.getUsed());
                assertEquals(5, fv.getNumImplicitEntries());
            }
            {
                FactorTemplate template = ofc.getTemplates().get(1);                
                ObsFeTypedFactor factor = ((ObsFeTypedFactor)fg.getFactor(1));
                assertEquals(2, template.getNumConfigs());
                assertEquals(Arrays.asList(SprlFactorType.SPRL_UNARY, "volitional"), factor.getTemplateKey());
                FeatureVector fv = builder.getFeatExtractor().calcObsFeatureVector(factor);
                assertEquals(5, fv.getUsed());
                assertEquals(5, fv.getNumImplicitEntries());
            }
            {
                FactorTemplate template = ofc.getTemplates().get(2);                
                ObsFeTypedFactor factor = ((ObsFeTypedFactor)fg.getFactor(2));
                assertEquals(Arrays.asList(SprlFactorType.SPRL_PAIRWISE, "volitional", "awareness"), factor.getTemplateKey());
                assertEquals(4, template.getNumConfigs());
                FeatureVector fv = builder.getFeatExtractor().calcObsFeatureVector(factor);
                // only bias
                // only one configuration is observed
                assertEquals(1, fv.getUsed());
                assertEquals(1, fv.getNumImplicitEntries());
            }
        }

        // sprlSrl factors
        SrlFactorGraphBuilderPrm srlPrm = new SrlFactorGraphBuilderPrm();
        srlPrm.allowPredArgSelfLoops = prm.allowPredArgSelfLoops;
        srlPrm.binarySenseRoleFactors = false;
        srlPrm.fcmFactors = false;
        srlPrm.makeUnknownPredRolesLatent = false;
        srlPrm.predictPredPos = false;
        srlPrm.roleStructure = prm.roleStructure;
        srlPrm.srlFePrm = srlFePrm;
        srlPrm.unaryFactors = true;
        {
            srlFePrm.argTemplates = Arrays.asList(new FeatTemplate1(Position.PARENT, PositionModifier.HEAD, TokProperty.CAPITALIZED));
            prm.srlFePrm = srlFePrm;
            prm.pairwiseFactors = false;
            CorpusStatistics cs = new CorpusStatistics(csPrm);
            ObsFeatureConjoiner ofc = new ObsFeatureConjoiner(ofcPrm, new FactorTemplateList());
            if (!cs.isInitialized()) {
                cs.init(sents, false);
                cs.store = new AlphabetStore(sents);
            }
            FgExampleListBuilder exListBuilder = new FgExampleListBuilder(exListPrm);
            FgExampleList exampleList = exListBuilder.getInstance(new FgExampleList() {
                
                @Override
                public int size() {
                    return sents.size();
                }
                
                @Override
                public LFgExample get(int i) {
                    SprlFactorGraphBuilder builder = new SprlFactorGraphBuilder(prm);
                    SrlFactorGraphBuilder srlBuilder = new SrlFactorGraphBuilder(srlPrm);
                    AnnoSentence sent = sents.get(i);
                    IntAnnoSentence isent = new IntAnnoSentence(sent, cs.store);
                    FactorGraph fg = new FactorGraph(); 
                    builder.build(isent, ofc, fg, cs);
                    VarConfig vc = new VarConfig();
                    builder.annoToConfig(sent, vc);
                    srlBuilder.build(isent, cs, ofc, fg);
                    SprlFactorGraphBuilder.addSprlSrlFactors(sent, ofc, cs, fg, builder, srlBuilder, prm.pairwiseFactors);
                    SrlEncoder.addSrlTrainAssignment(sent, sent.getSrlGraph(), srlBuilder, vc, srlPrm.predictSense,  srlPrm.predictPredPos,  srlPrm.predictPredPos);
                    return new LabeledFgExample(fg, vc, ofc.getTemplates());
                }
            });
            ofc.init(exampleList);
            LFgExample ex = exampleList.get(0);
            FactorGraph fg = ex.getFactorGraph();
            // 2 pairs each with (2 spr properties and 1 srl)
            assertEquals(6, fg.getNumVars());

            // 6 unary plus 4 role-spr pairwise plus two role,spr,spr three-way factor
            assertEquals(10, fg.getNumFactors());
            assertEquals(10, fg.getFactors().size());

            // 2 unary sprl 1 role 2 pairwise role-sprl, 1 three-way
            assertEquals(5, ofc.getTemplates().size());
            assertEquals(Arrays.asList(SprlFactorType.SPRL_UNARY, "awareness"), ofc.getTemplates().get(0).getKey());
            assertEquals(Arrays.asList(SprlFactorType.SPRL_UNARY, "volitional"), ofc.getTemplates().get(1).getKey());
            assertEquals(SrlFactorTemplate.ROLE_UNARY, ofc.getTemplates().get(2).getKey());
            assertEquals(Arrays.asList(JointFactorTemplate.ROLE_SPRL_BINARY, "awareness"), ofc.getTemplates().get(3).getKey());
            assertEquals(Arrays.asList(JointFactorTemplate.ROLE_SPRL_BINARY, "volitional"), ofc.getTemplates().get(4).getKey());
            assertEquals(2, ofc.getTemplates().get(0).getNumConfigs());
            assertEquals(2, ofc.getTemplates().get(1).getNumConfigs());
            // argUnk, _, ARG0, ARG1, ARG1-to
            assertEquals(5, ofc.getTemplates().get(2).getNumConfigs());
            assertEquals(5 * 2, ofc.getTemplates().get(3).getNumConfigs());
            assertEquals(5 * 2, ofc.getTemplates().get(4).getNumConfigs());
        }

        {
            // srl++sprl+
            prm.pairwiseFactors = true;
            srlFePrm.argTemplates = Arrays.asList(new FeatTemplate1(Position.PARENT, PositionModifier.HEAD, TokProperty.CAPITALIZED));
            prm.srlFePrm = srlFePrm;
            prm.pairwiseFactors = true;
            CorpusStatistics cs = new CorpusStatistics(csPrm);
            ObsFeatureConjoiner ofc = new ObsFeatureConjoiner(ofcPrm, new FactorTemplateList());
            if (!cs.isInitialized()) {
                cs.init(sents, false);
                cs.store = new AlphabetStore(sents);
            }
            FgExampleListBuilder exListBuilder = new FgExampleListBuilder(exListPrm);
            FgExampleList exampleList = exListBuilder.getInstance(new FgExampleList() {
                
                @Override
                public int size() {
                    return sents.size();
                }
                
                @Override
                public LFgExample get(int i) {
                    SprlFactorGraphBuilder builder = new SprlFactorGraphBuilder(prm);
                    SrlFactorGraphBuilder srlBuilder = new SrlFactorGraphBuilder(srlPrm);
                    AnnoSentence sent = sents.get(i);
                    IntAnnoSentence isent = new IntAnnoSentence(sent, cs.store);
                    FactorGraph fg = new FactorGraph(); 
                    builder.build(isent, ofc, fg, cs);
                    VarConfig vc = new VarConfig();
                    builder.annoToConfig(sent, vc);
                    srlBuilder.build(isent, cs, ofc, fg);
                    SprlFactorGraphBuilder.addSprlSrlFactors(sent, ofc, cs, fg, builder, srlBuilder, prm.pairwiseFactors);
                    SrlEncoder.addSrlTrainAssignment(sent, sent.getSrlGraph(), srlBuilder, vc, srlPrm.predictSense,  srlPrm.predictPredPos,  srlPrm.predictPredPos);
                    return new LabeledFgExample(fg, vc, ofc.getTemplates());
                }
            });
            ofc.init(exampleList);
            LFgExample ex = exampleList.get(0);
            FactorGraph fg = ex.getFactorGraph();
            // 2 pairs each with 2 properties and 1 srl
            assertEquals(6, fg.getNumVars());

            // 6 unary plus,  4 rol-spr pairwise, 2 spr-spr pairwise 
            assertEquals(12, fg.getNumFactors());
            assertEquals(12, fg.getFactors().size());

            // 2 unary sprl 1 role 2 pairwise role-sprl 1 spr-spr
            assertEquals(6, ofc.getTemplates().size());
            assertEquals(Arrays.asList(SprlFactorType.SPRL_UNARY, "awareness"), ofc.getTemplates().get(0).getKey());
            assertEquals(Arrays.asList(SprlFactorType.SPRL_UNARY, "volitional"), ofc.getTemplates().get(1).getKey());
            assertEquals(Arrays.asList(SprlFactorType.SPRL_PAIRWISE, "volitional", "awareness"), ofc.getTemplates().get(2).getKey());
            assertEquals(SrlFactorTemplate.ROLE_UNARY, ofc.getTemplates().get(3).getKey());
            assertEquals(Arrays.asList(JointFactorTemplate.ROLE_SPRL_BINARY, "awareness"), ofc.getTemplates().get(4).getKey());
            assertEquals(Arrays.asList(JointFactorTemplate.ROLE_SPRL_BINARY, "volitional"), ofc.getTemplates().get(5).getKey());
            assertEquals(2, ofc.getTemplates().get(0).getNumConfigs());
            assertEquals(2, ofc.getTemplates().get(1).getNumConfigs());
            assertEquals(4, ofc.getTemplates().get(2).getNumConfigs());
            // argUnk, _, ARG0, ARG1, ARG1-to
            assertEquals(5, ofc.getTemplates().get(3).getNumConfigs());
            assertEquals(5 * 2, ofc.getTemplates().get(4).getNumConfigs());
            assertEquals(5 * 2, ofc.getTemplates().get(5).getNumConfigs());
        }

        {
            // srlGsprl+
            prm.pairwiseFactors = true;
            srlFePrm.argTemplates = Arrays.asList(new FeatTemplate1(Position.PARENT, PositionModifier.HEAD, TokProperty.CAPITALIZED));
            prm.srlFePrm = srlFePrm;
            prm.pairwiseFactors = true;
            CorpusStatistics cs = new CorpusStatistics(csPrm);
            ObsFeatureConjoiner ofc = new ObsFeatureConjoiner(ofcPrm, new FactorTemplateList());
            if (!cs.isInitialized()) {
                cs.init(sents, false);
                cs.store = new AlphabetStore(sents);
            }
            FgExampleListBuilder exListBuilder = new FgExampleListBuilder(exListPrm);
            FgExampleList exampleList = exListBuilder.getInstance(new FgExampleList() {
                
                @Override
                public int size() {
                    return sents.size();
                }
                
                @Override
                public LFgExample get(int i) {
                    SrlFactorGraphBuilder srlBuilder = new SrlFactorGraphBuilder(srlPrm);
                    AnnoSentence sent = sents.get(i);
                    IntAnnoSentence isent = new IntAnnoSentence(sent, cs.store);
                    FactorGraph fg = new FactorGraph(); 
                    VarConfig vc = new VarConfig();
                    srlBuilder.build(isent, cs, ofc, fg);
                    SprlFactorGraphBuilder.addSprlSrlFactors(sent, ofc, cs, fg, null, srlBuilder, prm.pairwiseFactors);
                    SrlEncoder.addSrlTrainAssignment(sent, sent.getSrlGraph(), srlBuilder, vc, srlPrm.predictSense,  srlPrm.predictPredPos,  srlPrm.predictPredPos);
                    return new LabeledFgExample(fg, vc, ofc.getTemplates());
                }
            });
            ofc.init(exampleList);
            LFgExample ex = exampleList.get(0);
            FactorGraph fg = ex.getFactorGraph();
            // 2 role variables
            assertEquals(2, fg.getNumVars());

            // 2 * (1 role unary; 2 spr unaries; 1 spr pair)
            assertEquals(8, fg.getNumFactors());
            assertEquals(8, fg.getFactors().size());

            // role + role|aw=un + role|vol=un + role|vol=un&role|aw=un + role|aw=lk + role|vol=lk + role|vol=lk&role|aw=lk 
            // 7
            // order should be sorted by pred,arg,property
            assertEquals(7, ofc.getTemplates().size());
            assertEquals(SrlFactorTemplate.ROLE_UNARY, ofc.getTemplates().get(0).getKey());
            assertEquals(Arrays.asList(JointFactorTemplate.ROLE_SPRL_BINARY, "GOLD_SPRL", "awareness", SprlLabelConverter.LIKELY), ofc.getTemplates().get(1).getKey());
            assertEquals(Arrays.asList(JointFactorTemplate.ROLE_SPRL_BINARY, "GOLD_SPRL", "volitional", SprlLabelConverter.LIKELY), ofc.getTemplates().get(2).getKey());
            assertEquals(Arrays.asList(JointFactorTemplate.ROLE_SPRL_SPRL, "GOLD_SPRL_PAIR", "volitional", SprlLabelConverter.LIKELY, "awareness", SprlLabelConverter.LIKELY), ofc.getTemplates().get(3).getKey());
            assertEquals(Arrays.asList(JointFactorTemplate.ROLE_SPRL_BINARY, "GOLD_SPRL", "awareness", SprlLabelConverter.UNLIKELY), ofc.getTemplates().get(4).getKey());
            assertEquals(Arrays.asList(JointFactorTemplate.ROLE_SPRL_BINARY, "GOLD_SPRL", "volitional", SprlLabelConverter.UNLIKELY), ofc.getTemplates().get(5).getKey());
            assertEquals(Arrays.asList(JointFactorTemplate.ROLE_SPRL_SPRL, "GOLD_SPRL_PAIR", "volitional", SprlLabelConverter.UNLIKELY, "awareness", SprlLabelConverter.UNLIKELY), ofc.getTemplates().get(6).getKey());
            for (int i = 0; i < 7; i++) {
                // argUnk, _, ARG0, ARG1, ARG1-to
                assertEquals(5, ofc.getTemplates().get(i).getNumConfigs());
            }
        }

        {
            // sprl+Gsrl
            prm.pairwiseFactors = true;
            srlFePrm.argTemplates = Arrays.asList(new FeatTemplate1(Position.PARENT, PositionModifier.HEAD, TokProperty.CAPITALIZED));
            prm.srlFePrm = srlFePrm;
            prm.pairwiseFactors = true;
            CorpusStatistics cs = new CorpusStatistics(csPrm);
            ObsFeatureConjoiner ofc = new ObsFeatureConjoiner(ofcPrm, new FactorTemplateList());
            if (!cs.isInitialized()) {
                cs.init(sents, false);
                cs.store = new AlphabetStore(sents);
            }
            FgExampleListBuilder exListBuilder = new FgExampleListBuilder(exListPrm);
            FgExampleList exampleList = exListBuilder.getInstance(new FgExampleList() {
                
                @Override
                public int size() {
                    return sents.size();
                }
                
                @Override
                public LFgExample get(int i) {
                    SprlFactorGraphBuilder builder = new SprlFactorGraphBuilder(prm);
                    AnnoSentence sent = sents.get(i);
                    IntAnnoSentence isent = new IntAnnoSentence(sent, cs.store);
                    FactorGraph fg = new FactorGraph(); 
                    builder.build(isent, ofc, fg, cs);
                    VarConfig vc = new VarConfig();
                    builder.annoToConfig(sent, vc);
                    SprlFactorGraphBuilder.addSprlSrlFactors(sent, ofc, cs, fg, builder, null, prm.pairwiseFactors);
                    return new LabeledFgExample(fg, vc, ofc.getTemplates());
                }
            });
            ofc.init(exampleList);
            LFgExample ex = exampleList.get(0);
            FactorGraph fg = ex.getFactorGraph();
            // 2 pairs with 2 prop vars each
            assertEquals(4, fg.getNumVars());

            // 4 vars * (1 prop unary + 1 gold role unary) + 2 pairs * (1 spr pairwise)
            assertEquals(10, fg.getNumFactors());
            assertEquals(10, fg.getFactors().size());

            // aw, vol, vol+aw, aware|role_sprl * 3 (observed), vol|role_sprl * 3 (observed in order observed)
            assertEquals(9, ofc.getTemplates().size());
            assertEquals(Arrays.asList(SprlFactorType.SPRL_UNARY, "awareness"), ofc.getTemplates().get(0).getKey());
            assertEquals(Arrays.asList(SprlFactorType.SPRL_UNARY, "volitional"), ofc.getTemplates().get(1).getKey());
            assertEquals(Arrays.asList(SprlFactorType.SPRL_PAIRWISE, "volitional", "awareness"), ofc.getTemplates().get(2).getKey());
            assertEquals(Arrays.asList(JointFactorTemplate.ROLE_SPRL_BINARY, "awareness", "GOLD_SRL", "ARG0-to"), ofc.getTemplates().get(3).getKey());
            assertEquals(Arrays.asList(JointFactorTemplate.ROLE_SPRL_BINARY, "volitional", "GOLD_SRL", "ARG0-to"), ofc.getTemplates().get(4).getKey());
            assertEquals(Arrays.asList(JointFactorTemplate.ROLE_SPRL_BINARY, "awareness", "GOLD_SRL", "ARG1"), ofc.getTemplates().get(5).getKey());
            assertEquals(Arrays.asList(JointFactorTemplate.ROLE_SPRL_BINARY, "volitional", "GOLD_SRL", "ARG1"), ofc.getTemplates().get(6).getKey());
            assertEquals(Arrays.asList(JointFactorTemplate.ROLE_SPRL_BINARY, "awareness", "GOLD_SRL", "ARG0"), ofc.getTemplates().get(7).getKey());
            assertEquals(Arrays.asList(JointFactorTemplate.ROLE_SPRL_BINARY, "volitional", "GOLD_SRL", "ARG0"), ofc.getTemplates().get(8).getKey());
            for (int i = 0; i < 7; i++) {
                if (i == 2) {
                    // the single pairwise template
                    assertEquals(4, ofc.getTemplates().get(i).getNumConfigs());
                } else {
                    assertEquals(2, ofc.getTemplates().get(i).getNumConfigs());
                }
            }
        }
        

        {
            // srlGsprl
            prm.pairwiseFactors = true;
            srlFePrm.argTemplates = Arrays.asList(new FeatTemplate1(Position.PARENT, PositionModifier.HEAD, TokProperty.CAPITALIZED));
            prm.srlFePrm = srlFePrm;
            prm.pairwiseFactors = false;
            CorpusStatistics cs = new CorpusStatistics(csPrm);
            ObsFeatureConjoiner ofc = new ObsFeatureConjoiner(ofcPrm, new FactorTemplateList());
            if (!cs.isInitialized()) {
                cs.init(sents, false);
                cs.store = new AlphabetStore(sents);
            }
            FgExampleListBuilder exListBuilder = new FgExampleListBuilder(exListPrm);
            FgExampleList exampleList = exListBuilder.getInstance(new FgExampleList() {
                
                @Override
                public int size() {
                    return sents.size();
                }
                
                @Override
                public LFgExample get(int i) {
                    SrlFactorGraphBuilder srlBuilder = new SrlFactorGraphBuilder(srlPrm);
                    AnnoSentence sent = sents.get(i);
                    IntAnnoSentence isent = new IntAnnoSentence(sent, cs.store);
                    FactorGraph fg = new FactorGraph(); 
                    VarConfig vc = new VarConfig();
                    srlBuilder.build(isent, cs, ofc, fg);
                    SprlFactorGraphBuilder.addSprlSrlFactors(sent, ofc, cs, fg, null, srlBuilder, prm.pairwiseFactors);
                    SrlEncoder.addSrlTrainAssignment(sent, sent.getSrlGraph(), srlBuilder, vc, srlPrm.predictSense,  srlPrm.predictPredPos,  srlPrm.predictPredPos);
                    return new LabeledFgExample(fg, vc, ofc.getTemplates());
                }
            });
            ofc.init(exampleList);
            LFgExample ex = exampleList.get(0);
            FactorGraph fg = ex.getFactorGraph();
            // 2 role variables
            assertEquals(2, fg.getNumVars());

            // 2 * (1 role unary; 2 spr unaries)
            assertEquals(6, fg.getNumFactors());
            assertEquals(6, fg.getFactors().size());

            // role + role|aw=un + role|vol=un + role|aw=lk + role|vol=lk 
            assertEquals(5, ofc.getTemplates().size());
            assertEquals(SrlFactorTemplate.ROLE_UNARY, ofc.getTemplates().get(0).getKey());
            assertEquals(Arrays.asList(JointFactorTemplate.ROLE_SPRL_BINARY, "GOLD_SPRL", "awareness", SprlLabelConverter.LIKELY), ofc.getTemplates().get(1).getKey());
            assertEquals(Arrays.asList(JointFactorTemplate.ROLE_SPRL_BINARY, "GOLD_SPRL", "volitional", SprlLabelConverter.LIKELY), ofc.getTemplates().get(2).getKey());
            assertEquals(Arrays.asList(JointFactorTemplate.ROLE_SPRL_BINARY, "GOLD_SPRL", "awareness", SprlLabelConverter.UNLIKELY), ofc.getTemplates().get(3).getKey());
            assertEquals(Arrays.asList(JointFactorTemplate.ROLE_SPRL_BINARY, "GOLD_SPRL", "volitional", SprlLabelConverter.UNLIKELY), ofc.getTemplates().get(4).getKey());
            for (int i = 0; i < 5; i++) {
                // argUnk, _, ARG0, ARG1, ARG1-to
                assertEquals(5, ofc.getTemplates().get(i).getNumConfigs());
            }
        }

        {
            // neither given
            prm.pairwiseFactors = true;
            srlFePrm.argTemplates = Arrays.asList(new FeatTemplate1(Position.PARENT, PositionModifier.HEAD, TokProperty.CAPITALIZED));
            prm.srlFePrm = srlFePrm;
            prm.pairwiseFactors = false;
            CorpusStatistics cs = new CorpusStatistics(csPrm);
            ObsFeatureConjoiner ofc = new ObsFeatureConjoiner(ofcPrm, new FactorTemplateList());
            if (!cs.isInitialized()) {
                cs.init(sents, false);
                cs.store = new AlphabetStore(sents);
            }
            AnnoSentence sent = sents.get(0);
            SrlFactorGraphBuilder srlBuilder = new SrlFactorGraphBuilder(srlPrm);
            {
                FactorGraph fg = new FactorGraph(); 
                SprlFactorGraphBuilder.addSprlSrlFactors(sent, ofc, cs, fg, null, null, prm.pairwiseFactors);
                assertEquals(0, fg.getNumVars());
                assertEquals(0, fg.getNumFactors());
            }
            {
                FactorGraph fg = new FactorGraph(); 
                prm.roleStructure = RoleStructure.ALL_PAIRS;
                SprlFactorGraphBuilder builder = new SprlFactorGraphBuilder(prm);
                assertTrue(TestUtils.checkThrows(() -> SprlFactorGraphBuilder.addSprlSrlFactors(sent, ofc, cs, fg, builder, srlBuilder, prm.pairwiseFactors), IllegalArgumentException.class));
                assertEquals(0, fg.getNumVars());
                assertEquals(0, fg.getNumFactors());
            }
            {
                FactorGraph fg = new FactorGraph(); 
                prm.roleStructure = RoleStructure.PAIRS_GIVEN;
                prm.allowPredArgSelfLoops = false;
                SprlFactorGraphBuilder builder = new SprlFactorGraphBuilder(prm);
                assertTrue(TestUtils.checkThrows(() -> SprlFactorGraphBuilder.addSprlSrlFactors(sent, ofc, cs, fg, builder, srlBuilder, prm.pairwiseFactors), IllegalArgumentException.class));
                assertEquals(0, fg.getNumVars());
                assertEquals(0, fg.getNumFactors());
            }
        }
        
    }

}
