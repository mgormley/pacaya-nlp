package edu.jhu.nlp.sprl;

import static org.junit.Assert.assertEquals;

import java.io.File;
import java.io.IOException;

import org.junit.Test;

import edu.jhu.nlp.data.concrete.ConcreteReader;
import edu.jhu.nlp.data.concrete.ConcreteReader.ConcreteReaderPrm;
import edu.jhu.nlp.data.simple.AnnoSentenceCollection;

public class SprlFactorGraphBuilderTest {
    private static String concreteFilename = "/edu/jhu/nlp/data/concrete/sprlmini.comm";

    @Test
    public void testCounts() throws IOException {
        File f = new File(getClass().getResource(concreteFilename).getFile());
        ConcreteReaderPrm prm = new ConcreteReaderPrm();
        prm.srlTool = "fpropbank";
        prm.sprlTool = "fpropbank";
        ConcreteReader r = new ConcreteReader(prm);
        AnnoSentenceCollection sents = r.sentsFromCommFile(f);
        assertEquals(2, sents.size());
    }

}
