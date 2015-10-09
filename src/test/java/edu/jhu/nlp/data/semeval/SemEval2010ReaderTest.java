package edu.jhu.nlp.data.semeval;

import static org.junit.Assert.assertEquals;

import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;

import org.junit.Test;

import edu.jhu.nlp.data.Span;
import edu.jhu.pacaya.util.collections.QLists;


public class SemEval2010ReaderTest {

    String s1 = "1\t\"<e1>The man</e1> jumped over <e2>the moon</e2> .\"\n"
            + "Was-Jumped(e2,e1)\n"
            + "Comment: He jumped very high.\n"
            + "\n";

    String s2 = "2\t\"The <e2><e1>man</e1></e2> jumped over the moon .\"\n"
            + "Self(e1,e2)\n"
            + "Comment: He jumped very high.\n"
            + "\n";

    String s3 = "2\t\"the <e1><e2>extra crispy</e1></e2> chip\"\n"
            + "Self(e1,e2)\n"
            + "Comment: He jumped very high.\n"
            + "\n";
        
    @Test
    public void testReadSentence1() throws Exception {
        InputStream stream = new ByteArrayInputStream(s1.getBytes(StandardCharsets.UTF_8));
        SemEval2010Reader reader = new SemEval2010Reader(stream);
        SemEval2010Sentence sent = reader.next();
        assertEquals(QLists.getList("The", "man", "jumped", "over", "the", "moon", "."), sent.words);
        assertEquals(new Span(0, 2), sent.e1.getSpan());
        assertEquals(new Span(4, 6), sent.e2.getSpan());
        reader.close();
    }

    @Test
    public void testReadSentence2() throws Exception {
        InputStream stream = new ByteArrayInputStream(s2.getBytes(StandardCharsets.UTF_8));
        SemEval2010Reader reader = new SemEval2010Reader(stream);
        SemEval2010Sentence sent = reader.next();
        assertEquals(QLists.getList("The", "man", "jumped", "over", "the", "moon", "."), sent.words);
        assertEquals(new Span(1, 2), sent.e1.getSpan());
        assertEquals(new Span(1, 2), sent.e2.getSpan());
        reader.close();
    }
    
    @Test
    public void testReadSentence3() throws Exception {
        InputStream stream = new ByteArrayInputStream(s3.getBytes(StandardCharsets.UTF_8));
        SemEval2010Reader reader = new SemEval2010Reader(stream);
        SemEval2010Sentence sent = reader.next();
        assertEquals(QLists.getList("the", "extra", "crispy", "chip"), sent.words);
        assertEquals(new Span(1, 3), sent.e1.getSpan());
        assertEquals(new Span(1, 3), sent.e2.getSpan());
        reader.close();
    }
    
}
