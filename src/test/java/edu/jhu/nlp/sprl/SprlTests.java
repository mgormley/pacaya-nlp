package edu.jhu.nlp.sprl;

import org.junit.runner.RunWith;
import org.junit.runners.Suite;

@RunWith(Suite.class)
@Suite.SuiteClasses({
    ConfusionMatrixTest.class,
    ConfusionMapTest.class,
    CounterTest.class,
    BinarySprlLabelConverterTest.class,
    SprlPropertiesTest.class,
    SprlConcreteEvaluatorTest.class,
    SprlEvaluatorTest.class,
    SprlFactorGraphBuilderTest.class,
    SprlLearningTest.class
})
public class SprlTests { }
