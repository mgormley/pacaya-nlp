package edu.jhu.nlp.eval;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.nlp.Evaluator;
import edu.jhu.nlp.data.DepGraph;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.data.simple.AnnoSentenceCollection;

public class SrlSelfLoops implements Evaluator {
    
    private static final Logger log = LoggerFactory.getLogger(SrlSelfLoops.class);

    @Override
    public double evaluate(AnnoSentenceCollection predSents, AnnoSentenceCollection goldSents, String name) {
        return printPredArgSelfLoopStats(goldSents, "gold " + name);
    }
    
    private static double printPredArgSelfLoopStats(AnnoSentenceCollection sents, String name) {
        int numPredArgSelfLoop = 0;
        int numPredArgs = 0;
        for (AnnoSentence sent : sents) {
            DepGraph srl = sent.getSrlGraph();
            if (srl != null) {
                for (int p=-1; p<srl.size(); p++) {
                    for (int c=0; c<srl.size(); c++) {
                        if (srl.get(p, c) != null) {
                            if (p == c) { numPredArgSelfLoop++; }
                            numPredArgs++;
                        }
                    }
                }
            }
        }
        log.info(String.format("Proportion pred-arg self loops on %s: %.4f (%d / %d)", name, (double) numPredArgSelfLoop/numPredArgs, numPredArgSelfLoop, numPredArgs));
        return 0;
    }

}
