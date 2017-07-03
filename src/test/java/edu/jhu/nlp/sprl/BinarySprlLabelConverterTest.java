package edu.jhu.nlp.sprl;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static edu.jhu.pacaya.sch.util.TestUtils.checkThrows;
import org.junit.Test;

import edu.jhu.pacaya.sch.util.TestUtils;

public class BinarySprlLabelConverterTest {
    private static double tol = 1E-9;
    
    @Test
    public void test() {
        BinarySprlLabelConverter converter = new BinarySprlLabelConverter(4.0);
        assertEquals(SprlLabelConverter.UNLIKELY, converter.toLabel(3.0, true));
        assertEquals(SprlLabelConverter.UNLIKELY, converter.toLabel(3.0, false));
        assertEquals(SprlLabelConverter.UNLIKELY, converter.toLabel(4.0, false));
        assertEquals(SprlLabelConverter.UNLIKELY, converter.toLabel(5.0, false));
        assertEquals(SprlLabelConverter.LIKELY, converter.toLabel(4.0, true));
        assertEquals(SprlLabelConverter.LIKELY, converter.toLabel(5.0, true));
        assertEquals(SprlLabelConverter.UNLIKELY, converter.toLabel(3.0, 1.0));
        assertEquals(SprlLabelConverter.UNLIKELY, converter.toLabel(3.0, -1.0));
        assertEquals(SprlLabelConverter.UNLIKELY, converter.toLabel(4.0, -1.0));
        assertEquals(SprlLabelConverter.UNLIKELY, converter.toLabel(5.0, -1.0));
        assertEquals(SprlLabelConverter.LIKELY, converter.toLabel(4.0, 1.0));
        assertEquals(SprlLabelConverter.LIKELY, converter.toLabel(5.0, 1.0));
        assertTrue(TestUtils.checkThrows(() -> converter.toLabel(5.0, 0.0), IllegalArgumentException.class));
        assertEquals(true, converter.readApplicable(SprlLabelConverter.LIKELY));
        assertEquals(true, converter.readApplicable(SprlLabelConverter.UNLIKELY));
        assertTrue(checkThrows(() -> converter.readApplicable(SprlLabelConverter.UNKNOWN), IllegalArgumentException.class));
        assertEquals(5.0, converter.readProb(SprlLabelConverter.LIKELY), tol);
        assertEquals(1.0, converter.readProb(SprlLabelConverter.UNLIKELY), tol);
        assertTrue(checkThrows(() -> converter.readProb(SprlLabelConverter.UNKNOWN), IllegalArgumentException.class));
        assertTrue(converter.isNil(SprlLabelConverter.nil()));
    }

}
