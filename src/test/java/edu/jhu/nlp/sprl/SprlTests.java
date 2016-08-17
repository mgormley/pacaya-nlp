package edu.jhu.nlp.sprl;

import org.junit.runner.RunWith;
import org.junit.runners.Suite;

import edu.jhu.nlp.data.concrete.ConcreteReaderTest;

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
    SprlLearningTest.class,
    ConcreteReaderTest.class
})
public class SprlTests { }
