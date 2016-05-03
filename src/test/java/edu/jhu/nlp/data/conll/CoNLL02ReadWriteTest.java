package edu.jhu.nlp.data.conll;

import java.io.IOException;
import java.io.InputStream;
import java.io.StringWriter;

import org.junit.Assert;
import org.junit.Test;

import edu.jhu.pacaya.util.files.QFiles;

public class CoNLL02ReadWriteTest {

    public static final String conllXExample= "/edu/jhu/nlp/data/conll/conll-02-example.conll";
    
    @Test
    public void testReadWrite() throws IOException {
        InputStream inputStream = this.getClass().getResourceAsStream(conllXExample);
        CoNLL02Reader cr = new CoNLL02Reader(inputStream);

        StringWriter writer = new StringWriter();
        CoNLL02Writer cw = new CoNLL02Writer(writer);
        for (CoNLL02Sentence sent : cr) {
            cw.write(sent);
        }
        cw.close();
        cr.close();
        
        String readSentsStr = QFiles.getResourceAsString(conllXExample, "iso-8859-1");
        String writeSentsStr = writer.getBuffer().toString();
        String[] readSplits = readSentsStr.split("\n");
        String[] writeSplits = writeSentsStr.split("\n");
        
        // Check that it writes the correct number of lines.
        Assert.assertEquals(readSplits.length, writeSplits.length);
        for (int i=0; i<readSplits.length; i++) {
            System.out.println(readSplits[i]);
            System.out.println(writeSplits[i]);
            // Check that everything except for whitespace is identical.
            Assert.assertEquals(canonicalizeWhitespace(readSplits[i]), canonicalizeWhitespace(writeSplits[i]));    
        }
        // Check that whitespace is identical.
        Assert.assertEquals(readSentsStr, writeSentsStr);
    }

    private String canonicalizeWhitespace(String str) {
        return str.trim().replaceAll("[ \t]+", " ");
    }

}
