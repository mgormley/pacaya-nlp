package edu.jhu.nlp.data.simple;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.io.ByteArrayInputStream;

import org.junit.Test;

import edu.stanford.nlp.io.StringOutputStream;

public class JsonSentReaderTest {

    @Test
    public void testReadWriteSentence() throws Exception {
        ByteArrayInputStream bais = new ByteArrayInputStream(JsonSentWriterTest.expectedStr.getBytes("UTF-8"));
        JsonSentReader r = new JsonSentReader(bais);
        assertTrue(r.hasNext());
        AnnoSentence sent1 = r.next();
        assertTrue(r.hasNext());
        AnnoSentence sent2 = r.next();
        assertFalse(r.hasNext());
        r.close();

        // AnnoSentence expectedSent = JsonSentWriterTest.get6WordAnnoSentence();
        // expectedSent.setRelations(null);
        // expectedSent.setNamedEntities(null);
        
        StringOutputStream os = new StringOutputStream();
        JsonSentWriter w = new JsonSentWriter(os);
        w.write(sent1);
        w.write(sent2);
        w.close();
        String str = os.toString();
        
        System.out.println(str);
        assertEquals(JsonSentWriterTest.expectedStr, str);
    }

}
