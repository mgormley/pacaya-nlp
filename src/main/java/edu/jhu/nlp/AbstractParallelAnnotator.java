package edu.jhu.nlp;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.StringWriter;
import java.io.PrintWriter;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.data.simple.AnnoSentenceCollection;
import edu.jhu.pacaya.util.Threads;
import edu.jhu.prim.util.Lambda.FnIntToVoid;

public abstract class AbstractParallelAnnotator implements Annotator {

    private static final Logger log = LoggerFactory.getLogger(AbstractParallelAnnotator.class);
    private static final long serialVersionUID = 1L;
    // This is for debugging purposes; if you want a stack trace that
    // goes back to main, you can turn parallel off
    private boolean parallel = true;

    private void wrappedAnnotate(AnnoSentence sent) {
        try {
            annotate(sent);
        } catch (Throwable t) {
            AbstractParallelAnnotator.logThrowable(log, t);
        }
    }

    @Override
    public void annotate(final AnnoSentenceCollection sents) {
        // Add the new predictions to each sentence.
        if (parallel) {
            Threads.forEach(0, sents.size(), new FnIntToVoid() {            
                    @Override
                    public void call(int i) {
                        wrappedAnnotate(sents.get(i));
                    }
                });
        } else {
            for (int i = 0; i < sents.size(); i++) {
                wrappedAnnotate(sents.get(i));
            }
        }
    }

    public abstract void annotate(AnnoSentence sent);
    
    public static void logThrowable(Logger log, Throwable t) {
        StringWriter sW = new StringWriter();
        PrintWriter pW = new PrintWriter(sW);
        t.printStackTrace(pW);
        String msg = sW.getBuffer().toString();
        //String msg = (t.getMessage() == null) ? "" : ": " + t.getMessage();
        log.error("Failed to annotate sentence. Caught throwable: " + t.getClass() + msg);
        log.trace("Stacktrace from previous ERROR:\n", t);
    }
}
