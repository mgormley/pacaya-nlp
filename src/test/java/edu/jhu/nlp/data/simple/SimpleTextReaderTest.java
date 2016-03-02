package edu.jhu.nlp.data.simple;

import static org.junit.Assert.*;

import java.io.ByteArrayInputStream;

import org.junit.Test;

import edu.stanford.nlp.io.StringOutputStream;

public class SimpleTextReaderTest {

    @Test
    public void testReadWriteSentence() throws Exception {
        ByteArrayInputStream bais = new ByteArrayInputStream(SimpleTextWriterTest.expectedStr.getBytes("UTF-8"));
        SimpleTextReader r = new SimpleTextReader(bais);
        assertTrue(r.hasNext());
        AnnoSentence sent1 = r.next();
        assertTrue(r.hasNext());
        AnnoSentence sent2 = r.next();
        assertFalse(r.hasNext());
        r.close();

        // AnnoSentence expectedSent = SimpleTextWriterTest.get6WordAnnoSentence();
        // expectedSent.setRelations(null);
        // expectedSent.setNamedEntities(null);
        
        StringOutputStream os = new StringOutputStream();
        SimpleTextWriter w = new SimpleTextWriter(os);
        w.write(sent1);
        w.write(sent2);
        w.close();
        String str = os.toString();
        
        System.out.println(str);
        assertEquals(SimpleTextWriterTest.expectedStr, str);
    }

}
