package edu.jhu.nlp.features;

import static org.junit.Assert.assertEquals;

import java.util.ArrayList;
import java.util.List;

import org.junit.Test;

import edu.jhu.nlp.CorpusStatistics.CorpusStatisticsPrm;
import edu.jhu.nlp.data.conll.CoNLL09Sentence;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.data.simple.AnnoSentenceTest;
import edu.jhu.pacaya.parse.dep.ParentsArray;
import edu.jhu.prim.list.IntArrayList;
import edu.jhu.prim.tuple.Pair;

public class FeaturizedTokenPairTest {

    @Test
    public void testZhaoPathFeatures() {
        CoNLL09Sentence sent = getSpanishConll09Sentence2();
        CorpusStatisticsPrm csPrm = new CorpusStatisticsPrm();
        csPrm.useGoldSyntax = true;
        AnnoSentence simpleSent = sent.toAnnoSentence(csPrm.useGoldSyntax);

        FeaturizedToken zhaoPred = new FeaturizedToken(1, simpleSent);
        FeaturizedToken zhaoArg = new FeaturizedToken(0, simpleSent);
        FeaturizedTokenPair zhaoLink = new FeaturizedTokenPair(1, 0, zhaoPred, zhaoArg, simpleSent);
        List<Pair<Integer, ParentsArray.Dir>> desiredDpPathShare = new ArrayList<Pair<Integer, ParentsArray.Dir>>();
        desiredDpPathShare.add(new Pair<Integer, ParentsArray.Dir>(1, ParentsArray.Dir.UP));
        desiredDpPathShare.add(new Pair<Integer, ParentsArray.Dir>(-1, ParentsArray.Dir.NONE));
        List<Pair<Integer, ParentsArray.Dir>> observedDpPathShare = zhaoLink.getDpPathShare();
        System.out.println(observedDpPathShare);
        assertEquals(desiredDpPathShare, observedDpPathShare);
    }

    @Test
    public void testZhaoObjectPathSentence1() {
        CoNLL09Sentence sent = getSpanishConll09Sentence1();
        CorpusStatisticsPrm csPrm = new CorpusStatisticsPrm();
        csPrm.useGoldSyntax = true;
        AnnoSentence simpleSent = sent.toAnnoSentence(csPrm.useGoldSyntax);

        // Example indices.
        FeaturizedToken zhaoPred = new FeaturizedToken(3, simpleSent);
        FeaturizedToken zhaoArg = new FeaturizedToken(4, simpleSent);
        FeaturizedTokenPair zhaoLink = new FeaturizedTokenPair(3, 4, zhaoPred, zhaoArg, simpleSent);

        // Path between two indices.
        ArrayList<Pair<Integer, ParentsArray.Dir>> expectedPath = new ArrayList<Pair<Integer, ParentsArray.Dir>>();
        expectedPath.add(new Pair<Integer, ParentsArray.Dir>(3, ParentsArray.Dir.UP));
        expectedPath.add(new Pair<Integer, ParentsArray.Dir>(1, ParentsArray.Dir.DOWN));
        expectedPath.add(new Pair<Integer, ParentsArray.Dir>(4, ParentsArray.Dir.NONE));
        List<Pair<Integer, ParentsArray.Dir>> seenPath = zhaoLink.getDependencyPath();
        assertEquals(expectedPath, seenPath);

        // Shared path to root for two indices.
        List<Pair<Integer, ParentsArray.Dir>> dpPathShare = zhaoLink.getDpPathShare();
        ArrayList<Pair<Integer, ParentsArray.Dir>> expectedDpPathShare = new ArrayList<Pair<Integer, ParentsArray.Dir>>();
        expectedDpPathShare.add(new Pair<Integer, ParentsArray.Dir>(1, ParentsArray.Dir.UP));
        expectedDpPathShare.add(new Pair<Integer, ParentsArray.Dir>(-1, ParentsArray.Dir.NONE));
        assertEquals(dpPathShare, expectedDpPathShare);

        // New example indices.
        zhaoPred = new FeaturizedToken(0, simpleSent);
        zhaoArg = new FeaturizedToken(4, simpleSent);
        zhaoLink = new FeaturizedTokenPair(0, 4, zhaoPred, zhaoArg, simpleSent);

        // Path between two indices.
        expectedPath = new ArrayList<Pair<Integer, ParentsArray.Dir>>();
        expectedPath.add(new Pair<Integer, ParentsArray.Dir>(0, ParentsArray.Dir.UP));
        expectedPath.add(new Pair<Integer, ParentsArray.Dir>(1, ParentsArray.Dir.DOWN));
        expectedPath.add(new Pair<Integer, ParentsArray.Dir>(4, ParentsArray.Dir.NONE));
        seenPath = zhaoLink.getDependencyPath();
        assertEquals(expectedPath, seenPath);

        // Shared path to root for two indices.
        dpPathShare = zhaoLink.getDpPathShare();
        expectedDpPathShare = new ArrayList<Pair<Integer, ParentsArray.Dir>>();
        expectedDpPathShare.add(new Pair<Integer, ParentsArray.Dir>(1, ParentsArray.Dir.UP));
        expectedDpPathShare.add(new Pair<Integer, ParentsArray.Dir>(-1, ParentsArray.Dir.NONE));
        assertEquals(dpPathShare, expectedDpPathShare);

        // Line path (consecutive indices between two).
        IntArrayList linePath = zhaoLink.getLinePath();
        IntArrayList expectedLinePath = new IntArrayList();
        expectedLinePath.add(0);
        expectedLinePath.add(1);
        expectedLinePath.add(2);
        expectedLinePath.add(3);
        expectedLinePath.add(4);
        assertEquals(expectedLinePath, linePath);
    }

    @Test
    public void testZhaoObjectPathSentence2() {
        CoNLL09Sentence sent = getSpanishConll09Sentence2();
        CorpusStatisticsPrm csPrm = new CorpusStatisticsPrm();
        csPrm.useGoldSyntax = true;
        AnnoSentence simpleSent = sent.toAnnoSentence(csPrm.useGoldSyntax);

        // Example indices.
        FeaturizedToken zhaoPred = new FeaturizedToken(3, simpleSent);
        FeaturizedToken zhaoArg = new FeaturizedToken(4, simpleSent);
        FeaturizedTokenPair zhaoLink = new FeaturizedTokenPair(3, 4, zhaoPred, zhaoArg, simpleSent);

        // Path between two indices.
        ArrayList<Pair<Integer, ParentsArray.Dir>> expectedPath = new ArrayList<Pair<Integer, ParentsArray.Dir>>();
        expectedPath.add(new Pair<Integer, ParentsArray.Dir>(3, ParentsArray.Dir.UP));
        expectedPath.add(new Pair<Integer, ParentsArray.Dir>(5, ParentsArray.Dir.DOWN));
        expectedPath.add(new Pair<Integer, ParentsArray.Dir>(4, ParentsArray.Dir.NONE));
        List<Pair<Integer, ParentsArray.Dir>> seenPath = zhaoLink.getDependencyPath();
        assertEquals(expectedPath, seenPath);

        // Shared path to root for two indices.
        List<Pair<Integer, ParentsArray.Dir>> dpPathShare = zhaoLink.getDpPathShare();
        ArrayList<Pair<Integer, ParentsArray.Dir>> expectedDpPathShare = new ArrayList<Pair<Integer, ParentsArray.Dir>>();
        expectedDpPathShare.add(new Pair<Integer, ParentsArray.Dir>(5, ParentsArray.Dir.UP));
        expectedDpPathShare.add(new Pair<Integer, ParentsArray.Dir>(1, ParentsArray.Dir.UP));
        expectedDpPathShare.add(new Pair<Integer, ParentsArray.Dir>(-1, ParentsArray.Dir.NONE));
        assertEquals(dpPathShare, expectedDpPathShare);

        // New example indices.
        zhaoPred = new FeaturizedToken(0, simpleSent);
        zhaoArg = new FeaturizedToken(4, simpleSent);
        zhaoLink = new FeaturizedTokenPair(0, 4, zhaoPred, zhaoArg, simpleSent);

        // Path between two indices.
        expectedPath = new ArrayList<Pair<Integer, ParentsArray.Dir>>();
        expectedPath.add(new Pair<Integer, ParentsArray.Dir>(0, ParentsArray.Dir.UP));
        expectedPath.add(new Pair<Integer, ParentsArray.Dir>(1, ParentsArray.Dir.DOWN));
        expectedPath.add(new Pair<Integer, ParentsArray.Dir>(5, ParentsArray.Dir.DOWN));
        expectedPath.add(new Pair<Integer, ParentsArray.Dir>(4, ParentsArray.Dir.NONE));
        seenPath = zhaoLink.getDependencyPath();
        assertEquals(expectedPath, seenPath);

        // Shared path to root for two indices.
        dpPathShare = zhaoLink.getDpPathShare();
        expectedDpPathShare = new ArrayList<Pair<Integer, ParentsArray.Dir>>();
        expectedDpPathShare.add(new Pair<Integer, ParentsArray.Dir>(1, ParentsArray.Dir.UP));
        expectedDpPathShare.add(new Pair<Integer, ParentsArray.Dir>(-1, ParentsArray.Dir.NONE));
        assertEquals(dpPathShare, expectedDpPathShare);

        // Line path (consecutive indices between two).
        IntArrayList linePath = zhaoLink.getLinePath();
        IntArrayList expectedLinePath = new IntArrayList();
        expectedLinePath.add(0);
        expectedLinePath.add(1);
        expectedLinePath.add(2);
        expectedLinePath.add(3);
        expectedLinePath.add(4);
        assertEquals(expectedLinePath, linePath);
    }

    @Test
    public void testZhaoObjectPathSentence2PredictedSyntax() {
        CoNLL09Sentence sent = getSpanishConll09Sentence2();
        CorpusStatisticsPrm csPrm = new CorpusStatisticsPrm();
        csPrm.useGoldSyntax = false;
        AnnoSentence simpleSent = sent.toAnnoSentence(csPrm.useGoldSyntax);

        FeaturizedToken zhaoPred = new FeaturizedToken(3, simpleSent);
        FeaturizedToken zhaoArg = new FeaturizedToken(4, simpleSent);
        FeaturizedTokenPair zhaoLink = new FeaturizedTokenPair(3, 4, zhaoPred, zhaoArg, simpleSent);

        ArrayList<Pair<Integer, ParentsArray.Dir>> expectedPath = new ArrayList<Pair<Integer, ParentsArray.Dir>>();
        expectedPath.add(new Pair<Integer, ParentsArray.Dir>(3, ParentsArray.Dir.UP));
        expectedPath.add(new Pair<Integer, ParentsArray.Dir>(4, ParentsArray.Dir.NONE));
        List<Pair<Integer, ParentsArray.Dir>> seenPath = zhaoLink.getDependencyPath();
        assertEquals(expectedPath, seenPath);

        zhaoPred = new FeaturizedToken(0, simpleSent);
        zhaoArg = new FeaturizedToken(4, simpleSent);
        zhaoLink = new FeaturizedTokenPair(0, 4, zhaoPred, zhaoArg, simpleSent);

        expectedPath = new ArrayList<Pair<Integer, ParentsArray.Dir>>();
        expectedPath.add(new Pair<Integer, ParentsArray.Dir>(0, ParentsArray.Dir.DOWN));
        expectedPath.add(new Pair<Integer, ParentsArray.Dir>(6, ParentsArray.Dir.DOWN));
        expectedPath.add(new Pair<Integer, ParentsArray.Dir>(5, ParentsArray.Dir.DOWN));
        expectedPath.add(new Pair<Integer, ParentsArray.Dir>(4, ParentsArray.Dir.NONE));
        seenPath = zhaoLink.getDependencyPath();
        assertEquals(expectedPath, seenPath);
    }

    public static CoNLL09Sentence getSpanishConll09Sentence1() {
        return CoNLL09SentencesForTests.getSpanishConll09Sentence1();
    }

    public static CoNLL09Sentence getSpanishConll09Sentence2() {
        return CoNLL09SentencesForTests.getSpanishConll09Sentence2();
    }

    public static CoNLL09Sentence getDogConll09Sentence() {
        return AnnoSentenceTest.getDogConll09Sentence();
    }

}
