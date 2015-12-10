package edu.jhu.nlp.joint;

import static org.junit.Assert.fail;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectOutputStream;

import org.junit.Test;

import edu.jhu.nlp.CorpusStatistics;
import edu.jhu.nlp.CorpusStatistics.CorpusStatisticsPrm;
import edu.jhu.nlp.data.conll.CoNLL09Reader;
import edu.jhu.nlp.data.conll.CoNLL09ReadWriteTest;
import edu.jhu.nlp.data.conll.CoNLL09Sentence;
import edu.jhu.nlp.data.simple.AnnoSentenceCollection;
import edu.jhu.nlp.joint.JointNlpFactorGraph.JointNlpFactorGraphPrm;
import edu.jhu.pacaya.gm.feat.FactorTemplateList;
import edu.jhu.pacaya.gm.feat.ObsFeatureConjoiner;
import edu.jhu.pacaya.gm.feat.ObsFeatureConjoiner.ObsFeatureConjoinerPrm;
import edu.jhu.pacaya.gm.feat.ObsFeatureConjoinerTest;

public class JointNlpFgModelTest {

    @Test
    public void testIsSerializable() throws IOException {
        try {
            InputStream inputStream = this.getClass().getResourceAsStream(CoNLL09ReadWriteTest.conll2009Example);
            CoNLL09Reader cr = new CoNLL09Reader(inputStream);
            CorpusStatisticsPrm csPrm = new CorpusStatisticsPrm();
            AnnoSentenceCollection sents = CoNLL09Sentence.toAnno(cr.readSents(1), csPrm.useGoldSyntax);
            CorpusStatistics cs = new CorpusStatistics(csPrm);
            cs.init(sents);
            
            FactorTemplateList fts = ObsFeatureConjoinerTest.getFtl();
            ObsFeatureConjoinerPrm prm = new ObsFeatureConjoinerPrm();
            prm.featCountCutoff = -1;
            prm.includeUnsupportedFeatures = true;
            ObsFeatureConjoiner ofc = new ObsFeatureConjoiner(prm, fts);
            ofc.init(null);
            
            // Just test that no exception is thrown.
            JointNlpFgModel model = new JointNlpFgModel(cs, ofc, new JointNlpFactorGraphPrm());
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ObjectOutputStream out = new ObjectOutputStream(baos);
            out.writeObject(model);
            out.close();
        } catch(java.io.NotSerializableException e) {
            e.printStackTrace();
            fail("FgModel is not serializable: " + e.getMessage());
        }
    }

}
