package edu.jhu.nlp.data.simple;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;

import org.junit.Test;

public class JsonConcatReaderTest {

    @Test
    public void testReadWriteSentence() throws Exception {
        ByteArrayInputStream bais = new ByteArrayInputStream(JsonConcatWriterTest.expectedStr.getBytes("UTF-8"));
        JsonConcatReader r = new JsonConcatReader(bais);
        assertTrue(r.hasNext());
        AnnoSentence sent1 = r.next();
        assertTrue(r.hasNext());
        AnnoSentence sent2 = r.next();
        assertFalse(r.hasNext());
        r.close();

        // AnnoSentence expectedSent = JsonSentWriterTest.get6WordAnnoSentence();
        // expectedSent.setRelations(null);
        // expectedSent.setNamedEntities(null);
        
        ByteArrayOutputStream os = new ByteArrayOutputStream();
        JsonConcatWriter w = new JsonConcatWriter(os);
        w.write(sent1);
        w.write(sent2);
        w.close();
        String str = os.toString();
        
        System.out.println(str);
        assertEquals(JsonConcatWriterTest.expectedStr, str);
    }

}
