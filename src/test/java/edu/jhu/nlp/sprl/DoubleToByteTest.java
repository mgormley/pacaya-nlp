package edu.jhu.nlp.sprl;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

public class DoubleToByteTest {

    @Test
    public void testDoubleToByte() {
        assertEquals((byte)0, (byte)0.0);
        assertEquals((byte)1, (byte)1.0);
        assertEquals((byte)2, (byte)2.0);
        assertEquals((byte)3, (byte)3.0);
        assertEquals((byte)4, (byte)4.0);
        assertEquals((byte)5, (byte)5.0);
    }
    
    
}
