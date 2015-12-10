package edu.jhu.nlp;

import static org.junit.Assert.assertEquals;

import java.io.IOException;
import java.io.InputStream;
import java.util.HashSet;
import java.util.List;

import org.junit.Test;

import edu.jhu.nlp.CorpusStatistics.CorpusStatisticsPrm;
import edu.jhu.nlp.data.conll.CoNLL09Reader;
import edu.jhu.nlp.data.conll.CoNLL09ReadWriteTest;
import edu.jhu.nlp.data.conll.CoNLL09Sentence;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.data.simple.AnnoSentenceCollection;
import edu.jhu.pacaya.util.collections.QSets;

/**
 * Unit tests for {@link CorpusStatisticsTest}.
 * @author mgormley
 * @author mmitchell
 */
public class CorpusStatisticsTest {
    
    @Test
    public void testCreation() throws IOException {
        InputStream inputStream = this.getClass().getResourceAsStream(CoNLL09ReadWriteTest.conll2009Example);
        CoNLL09Reader cr = new CoNLL09Reader(inputStream);
        CorpusStatisticsPrm csPrm = new CorpusStatisticsPrm();
        csPrm.topN = 1;
        List<CoNLL09Sentence> sents = cr.readSents(4);
        cr.close();
        AnnoSentenceCollection simpleSents = new AnnoSentenceCollection();
        for (CoNLL09Sentence sent : sents) {
            sent.normalizeRoleNames();
            AnnoSentence simpleSent = sent.toAnnoSentence(csPrm.useGoldSyntax);
            simpleSents.add(simpleSent);
        }
        CorpusStatistics cs = new CorpusStatistics(csPrm);
        cs.init(simpleSents);

        assertEquals(QSets.getSet("de", ",", "."), new HashSet<String>(cs.knownWords));
        assertEquals(QSets.getSet("de"), new HashSet<String>(cs.topNWords));
        assertEquals(QSets.getSet("UNK-CAPS", "UNK", "UNK-LC", "UNK-LC-s"), new HashSet<String>(cs.knownUnks));
        assertEquals(QSets.getSet("True", "False"), new HashSet<String>(cs.linkStateNames));
        assertEquals(QSets.getSet("argUNK", "arg2", "arg1", "arg0", "argm", "_"), new HashSet<String>(cs.roleStateNames));
        assertEquals(30, cs.maxSentLength);        
    }

}
