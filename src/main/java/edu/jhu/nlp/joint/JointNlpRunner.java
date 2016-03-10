package edu.jhu.nlp.joint;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Date;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

import org.apache.commons.cli.ParseException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.hlt.optimize.Optimizer;
import edu.jhu.hlt.optimize.function.DifferentiableBatchFunction;
import edu.jhu.hlt.optimize.function.DifferentiableFunction;
import edu.jhu.hlt.optimize.function.Function;
import edu.jhu.nlp.AnnoPipeline;
import edu.jhu.nlp.Annotator;
import edu.jhu.nlp.CorpusStatistics.CorpusStatisticsPrm;
import edu.jhu.nlp.EvalPipeline;
import edu.jhu.nlp.TransientAnnotator;
import edu.jhu.nlp.data.Properties.Property;
import edu.jhu.nlp.data.simple.AnnoSentenceCollection;
import edu.jhu.nlp.data.simple.CorpusHandler;
import edu.jhu.nlp.data.simple.Experiments;
import edu.jhu.nlp.depparse.BitshiftDepParseFeatureExtractor.BitshiftDepParseFeatureExtractorPrm;
import edu.jhu.nlp.depparse.DepParseFactorGraphBuilder.DepParseFactorGraphBuilderPrm;
import edu.jhu.nlp.depparse.DepParseFeatureExtractor.DepParseFeatureExtractorPrm;
import edu.jhu.nlp.depparse.FirstOrderPruner;
import edu.jhu.nlp.depparse.GoldDepParseUnpruner;
import edu.jhu.nlp.depparse.O2AllGraFgInferencer.O2AllGraFgInferencerFactory;
import edu.jhu.nlp.depparse.PosTagDistancePruner;
import edu.jhu.nlp.embed.Embeddings;
import edu.jhu.nlp.embed.Embeddings.Scaling;
import edu.jhu.nlp.embed.EmbeddingsAnnotator;
import edu.jhu.nlp.embed.EmbeddingsAnnotator.EmbeddingsAnnotatorPrm;
import edu.jhu.nlp.eval.DepParseAccuracy;
import edu.jhu.nlp.eval.DepParseExactMatch;
import edu.jhu.nlp.eval.OraclePruningAccuracy;
import edu.jhu.nlp.eval.OraclePruningExactMatch;
import edu.jhu.nlp.eval.PosTagAccuracy;
import edu.jhu.nlp.eval.ProportionAnnotated;
import edu.jhu.nlp.eval.PruningEfficiency;
import edu.jhu.nlp.eval.RelationEvaluator;
import edu.jhu.nlp.eval.SprlEvaluator;
import edu.jhu.nlp.eval.SrlEvaluator;
import edu.jhu.nlp.eval.SrlEvaluator.SrlEvaluatorPrm;
import edu.jhu.nlp.eval.SrlPredIdAccuracy;
import edu.jhu.nlp.eval.SrlSelfLoops;
import edu.jhu.nlp.features.TemplateLanguage;
import edu.jhu.nlp.features.TemplateLanguage.AT;
import edu.jhu.nlp.features.TemplateLanguage.FeatTemplate;
import edu.jhu.nlp.features.TemplateLanguage.TokProperty;
import edu.jhu.nlp.features.TemplateReader;
import edu.jhu.nlp.features.TemplateSets;
import edu.jhu.nlp.joint.IGFeatureTemplateSelector.IGFeatureTemplateSelectorPrm;
import edu.jhu.nlp.joint.JointNlpAnnotator.InitParams;
import edu.jhu.nlp.joint.JointNlpAnnotator.JointNlpAnnotatorPrm;
import edu.jhu.nlp.joint.JointNlpDecoder.JointNlpDecoderPrm;
import edu.jhu.nlp.joint.JointNlpFgExamplesBuilder.JointNlpFgExampleBuilderPrm;
import edu.jhu.nlp.relations.RelationMunger;
import edu.jhu.nlp.relations.RelationMunger.RelationDataPostproc;
import edu.jhu.nlp.relations.RelationMunger.RelationDataPreproc;
import edu.jhu.nlp.relations.RelationMunger.RelationMungerPrm;
import edu.jhu.nlp.relations.RelationsFactorGraphBuilder.RelationsFactorGraphBuilderPrm;
import edu.jhu.nlp.sprl.SprlClassLabel;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.RoleStructure;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.SrlFactorGraphBuilderPrm;
import edu.jhu.nlp.srl.SrlFeatureExtractor.SrlFeatureExtractorPrm;
import edu.jhu.nlp.srl.SrlFeatureSelection;
import edu.jhu.nlp.srl.SrlWordFeatures;
import edu.jhu.nlp.srl.SrlWordFeatures.SrlWordFeaturesPrm;
import edu.jhu.nlp.tag.BrownClusterTagger;
import edu.jhu.nlp.tag.BrownClusterTagger.BrownClusterTaggerPrm;
import edu.jhu.nlp.tag.FileMapTagReducer;
import edu.jhu.nlp.tag.PosTagFactorGraphBuilder.PosTagFactorGraphBuilderPrm;
import edu.jhu.nlp.tag.StrictPosTagAnnotator;
import edu.jhu.nlp.words.PrefixAnnotator;
import edu.jhu.pacaya.gm.data.FgExampleListBuilder.CacheType;
import edu.jhu.pacaya.gm.decode.MbrDecoder.Loss;
import edu.jhu.pacaya.gm.decode.MbrDecoder.MbrDecoderPrm;
import edu.jhu.pacaya.gm.feat.ObsFeatureConjoiner.ObsFeatureConjoinerPrm;
import edu.jhu.pacaya.gm.inf.BeliefPropagation.BeliefPropagationPrm;
import edu.jhu.pacaya.gm.inf.BeliefPropagation.BpScheduleType;
import edu.jhu.pacaya.gm.inf.BeliefPropagation.BpUpdateOrder;
import edu.jhu.pacaya.gm.inf.BeliefsModuleFactory;
import edu.jhu.pacaya.gm.inf.BruteForceInferencer.BruteForceInferencerPrm;
import edu.jhu.pacaya.gm.inf.FgInferencerFactory;
import edu.jhu.pacaya.gm.model.Var.VarType;
import edu.jhu.pacaya.gm.train.CrfTrainer.CrfTrainerPrm;
import edu.jhu.pacaya.gm.train.CrfTrainer.Trainer;
import edu.jhu.pacaya.gm.train.DepParseSoftmaxMbr.DepParseSoftmaxMbrFactory;
import edu.jhu.pacaya.gm.train.ExpectedRecall.ExpectedRecallFactory;
import edu.jhu.pacaya.gm.train.L2Distance.L2DistanceFactory;
import edu.jhu.pacaya.hypergraph.depparse.InsideOutsideDepParse;
import edu.jhu.pacaya.util.Prm;
import edu.jhu.pacaya.util.Threads;
import edu.jhu.pacaya.util.cli.ArgParser;
import edu.jhu.pacaya.util.cli.Opt;
import edu.jhu.pacaya.util.collections.QSets;
import edu.jhu.pacaya.util.files.QFiles;
import edu.jhu.pacaya.util.report.Reporter;
import edu.jhu.pacaya.util.report.ReporterManager;
import edu.jhu.pacaya.util.semiring.Algebra;
import edu.jhu.pacaya.util.semiring.LogSemiring;
import edu.jhu.pacaya.util.semiring.LogSignAlgebra;
import edu.jhu.pacaya.util.semiring.RealAlgebra;
import edu.jhu.pacaya.util.semiring.ShiftedRealAlgebra;
import edu.jhu.pacaya.util.semiring.SplitAlgebra;
import edu.jhu.prim.tuple.Pair;
import edu.jhu.prim.util.Timer;
import edu.jhu.prim.util.math.FastMath;
import edu.jhu.prim.util.random.Prng;

/**
 * Pipeline runner for Joint NLP experiments.
 * @author mgormley
 * @author mmitchell
 */
public class JointNlpRunner {

    public enum ErmaLoss { L2DIST, EXPECTED_RECALL, SOFTMAX_MBR };

    public enum Inference { BRUTE_FORCE, BP, DP };

    public enum AlgebraType {
        REAL(RealAlgebra.getInstance()), LOG(LogSemiring.getInstance()), LOG_SIGN(LogSignAlgebra.getInstance()),
        // SHIFTED_REAL and SPLIT algebras are for testing only.
        SHIFTED_REAL(ShiftedRealAlgebra.getInstance()), SPLIT(SplitAlgebra.getInstance());

        private Algebra s;

        private AlgebraType(Algebra s) {
            this.s = s;
        }

        public Algebra getAlgebra() {
            return s;
        }
    }

    private static final Logger log = LoggerFactory.getLogger(JointNlpRunner.class);
    private static final Reporter rep = Reporter.getReporter(JointNlpRunner.class);

    // Options not specific to the model
    @Opt(name = "seed", hasArg = true, description = "Pseudo random number generator seed for everything else.")
    public static long seed = Prng.DEFAULT_SEED;
    @Opt(hasArg = true, description = "Number of threads for computation.")
    public static int threads = 1;
    @Opt(hasArg = true, description = "Whether to use a log-add table for faster computation.")
    public static boolean useLogAddTable = false;

    // Options for model IO
    @Opt(hasArg = true, description = "File from which to read a serialized model.")
    public static File modelIn = null;
    @Opt(hasArg = true, description = "File to which to serialize the model.")
    public static File modelOut = null;
    @Opt(hasArg = true, description = "File to which to print a human readable version of the model.")
    public static File printModel = null;
    @Opt(hasArg = true, description = "File to which to serialize the entire pipeline.")
    public static File pipeOut = null;

    // Options for joint model.
    @Opt(hasArg = true, description = "Whether to include the joint model in the pipeline.")
    public static boolean jointModel = true;

    // Options for initialization.
    @Opt(hasArg = true, description = "How to initialize the parameters of the model.")
    public static InitParams initParams = InitParams.UNIFORM;

    // Options for inference.
    @Opt(hasArg = true, description = "Type of inference method.")
    public static Inference inference = Inference.BP;
    @Opt(hasArg = true, description = "The algebra or semiring in which to run inference.")
    public static AlgebraType algebra = AlgebraType.LOG;
    @Opt(hasArg = true, description = "The BP schedule type.")
    public static BpScheduleType bpSchedule = BpScheduleType.TREE_LIKE;
    @Opt(hasArg = true, description = "The BP update order.")
    public static BpUpdateOrder bpUpdateOrder = BpUpdateOrder.SEQUENTIAL;
    @Opt(hasArg = true, description = "The max number of BP iterations.")
    public static int bpMaxIterations = 1;
    @Opt(hasArg = true, description = "Whether to normalize the messages.")
    public static boolean normalizeMessages = false;
    @Opt(hasArg = true, description = "The maximum message residual for convergence testing.")
    public static double bpConvergenceThreshold = 1e-3;
    @Opt(hasArg = true, description = "Directory to dump debugging information for BP.")
    public static File bpDumpDir = null;

    // Options for Brown clusters.
    @Opt(hasArg = true, description = "Brown cluster file")
    public static File brownClusters = null;
    @Opt(hasArg = true, description = "Max length for the brown clusters")
    public static int bcMaxTagLength = Integer.MAX_VALUE;

    // Options for Tag Maps
    @Opt(hasArg = true, description = "Type or file indicating tag mapping")
    public static File reduceTags = null;

    // Options for Embeddings.
    @Opt(hasArg=true, description="Path to word embeddings text file.")
    public static File embeddingsFile = null;
    @Opt(hasArg=true, description="Method for normalization of the embeddings.")
    public static Scaling embNorm = Scaling.L2_NORM;
    @Opt(hasArg=true, description="Amount to scale embeddings after normalization.")
    public static double embScalar = 1.0;
    @Opt(hasArg=true, description="Whether to use entity mention specific embeddings.")
    public static boolean entitySpecificEmbeddings = false;

    // Options for SRL factor graph structure.
    @Opt(hasArg = true, description = "The structure of the Role variables.")
    public static RoleStructure roleStructure = RoleStructure.PREDS_GIVEN;
    @Opt(hasArg = true, description = "Whether Role variables with unknown predicates should be latent.")
    public static boolean makeUnknownPredRolesLatent = true;
    @Opt(hasArg = true, description = "Whether to allow a predicate to assign a role to itself. (This should be turned on for English)")
    public static boolean allowPredArgSelfLoops = false;
    @Opt(hasArg = true, description = "Whether to include factors between the sense and role variables.")
    public static boolean binarySenseRoleFactors = false;
    @Opt(hasArg = true, description = "Whether to predict predicate sense.")
    public static boolean predictSense = false;
    @Opt(hasArg = true, description = "Whether to predict predicate positions.")
    public static boolean predictPredPos = false;
    @Opt(hasArg = true, description = " Whether to use FCM factors.")
    public static boolean srlFcmFactors = false;
    @Opt(hasArg = true, description = "Whether to treat the embeddings as model parameters.")
    public static boolean srlFcmFineTuning = false;
    @Opt(hasArg = true, description = "Whether to include unary factors on the SRL variables.")
    public static boolean srlUnaryFactors = true;

    // Options for SRL feature selection.
    @Opt(hasArg = true, description = "Whether to do feature selection.")
    public static boolean featureSelection = true;
    @Opt(hasArg = true, description = "The number of feature bigrams to form.")
    public static int numFeatsToSelect = 32;
    @Opt(hasArg = true, description = "The max number of sentences to use for feature selection")
    public static int numSentsForFeatSelect = 1000;

    // Options for feature extraction.
    @Opt(hasArg = true, description = "For testing only: whether to use only the bias feature.")
    public static boolean biasOnly = false;
    @Opt(hasArg = true, description = "The value of the mod for use in the feature hashing trick. If <= 0, feature-hashing will be disabled.")
    public static int featureHashMod = 524288; // 2^19

    // Options for SRL feature extraction.
    @Opt(hasArg = true, description = "Cutoff for OOV words.")
    public static int cutoff = 3;
    @Opt(hasArg = true, description = "For preprocessing: Minimum feature count for caching.")
    public static int featCountCutoff = 1;
    @Opt(hasArg = true, description = "Whether to include pairs of features.")
    public static boolean useTemplates = false;
    @Opt(hasArg = true, description = "Sense feature templates.")
    public static String senseFeatTpls = TemplateSets.bjorkelundSenseFeatsResource;
    @Opt(hasArg = true, description = "Arg feature templates.")
    public static String argFeatTpls = TemplateSets.bjorkelundArgFeatsResource;
    @Opt(hasArg = true, description = "Sense feature template output file.")
    public static File senseFeatTplsOut = null;
    @Opt(hasArg = true, description = "Arg feature template output file.")
    public static File argFeatTplsOut = null;
    @Opt(hasArg = true, description = "Whether to include extra ACL '14 style arg features.")
    public static boolean srlExtraArgFeats = false;
    @Opt(hasArg = true, description = "The type of the SRL role / sense variables.")
    public static VarType srlVarType = VarType.LATENT;
    @Opt(hasArg = true, description = "Whether to include pairwise factors between srl and sprl. If either sprl or srl are not included, then these become unary factors that are tied to the gold labels that are not being predicted.")
    public static boolean sprlSrlFactors = false;
    @Opt(hasArg = true, description = "Whether to include pairwise factors between all sprl questions for a given pred-arg pair.")
    public static boolean sprlAllPairs = false;
    @Opt(hasArg = true, description = "Whether to (add latent variable and pairwise factor)/(modify sprlSrlFactors) to includ the hard constraint that all sprl responses for a given pred-arg pair must agree as to whether or not it is an arg for the pred")
    public static boolean enforceSprlNilAgreement = true;
    @Opt(hasArg = true, description = "Whether to evaluate each sprl property separately")
    public static boolean breakdownSprlEval = true;
    @Opt(hasArg = true, description = "Only look at srl in validation function for srl with sprl")
    public static boolean favorSrlValidation = false;
    @Opt(hasArg = true, description = "Only look at sprl in validation function for srl with sprl")
    public static boolean favorSprlValidation = false;
    @Opt(hasArg = true, description = "If > 0, then only predict the ith property using observed features of the previous properties")
    public static int sprlPipelineIndex = -1;
    // the order of the sprl questions
    //public static List<Property> sprlPipelineOrder = Arrays.asList(Property.values());

    // TODO: add different order of property prediction
    // TODO: have the corpus statistics figure out what the properties instead of having an enum; that way things
    // still work when the SPRL questions change

    // Options for POS tagging factor graph structure.
    @Opt(hasArg = true, description = "The type of the tag variables.")
    public static VarType posTagVarType = VarType.LATENT;

    // Options for dependency parse factor graph structure.
    @Opt(hasArg = true, description = "The type of the link variables.")
    public static VarType linkVarType = VarType.LATENT;
    @Opt(hasArg = true, description = "Whether to include a projective dependency tree global factor.")
    public static boolean useProjDepTreeFactor = false;
    @Opt(hasArg = true, description = "Whether to include 2nd-order grandparent factors in the model.")
    public static boolean grandparentFactors = false;
    @Opt(hasArg = true, description = "Whether to include 2nd-order sibling factors in the model.")
    public static boolean arbitrarySiblingFactors = false;
    @Opt(hasArg = true, description = "Whether to include 2nd-order head-bigram factors in the model.")
    public static boolean headBigramFactors = false;
    @Opt(hasArg = true, description = "Whether to exclude non-projective grandparent factors.")
    public static boolean excludeNonprojectiveGrandparents = true;
    @Opt(hasArg = true, description = "Whether to include unary factors on the dependency edge variables.")
    public static boolean dpUnaryFactors = true;

    // Options for dependency parsing pruning.
    @Opt(hasArg = true, description = "File from which to read a first-order pruning model.")
    public static File pruneModel = null;
    @Opt(hasArg = true, description = "Whether to prune higher-order factors via a first-order pruning model.")
    public static boolean pruneByModel = false;
    @Opt(hasArg = true, description = "Whether to prune edges with a deterministic distance-based pruning approach.")
    public static boolean pruneByDist = false;

    // Options for Dependency parser feature extraction.
    @Opt(hasArg = true, description = "1st-order factor feature templates.")
    public static String dp1FeatTpls = TemplateSets.mcdonaldDepFeatsResource;
    @Opt(hasArg = true, description = "2nd-order factor feature templates.")
    public static String dp2FeatTpls = TemplateSets.carreras07Dep2FeatsResource;
    @Opt(hasArg = true, description = "Whether to use SRL features for dep parsing.")
    public static boolean acl14DepFeats = true;
    @Opt(hasArg = true, description = "Whether to use the fast feature set for dep parsing.")
    public static boolean dpFastFeats = true;

    @Opt(hasArg = true, description = "Whether to skip features that look at lemma.")
    public static boolean noLemma = false;
    @Opt(hasArg = true, description = "Whether to skip features that look at morpho feats.")
    public static boolean noMorpho = false;

    // Options for caching.
    @Opt(hasArg = true, description = "The type of cache/store to use for training/testing instances.")
    public static CacheType cacheType = CacheType.NONE;
    @Opt(hasArg = true, description = "When caching, the maximum number of examples to keep cached in memory or -1 for SoftReference caching.")
    public static int maxEntriesInMemory = 100;
    @Opt(hasArg = true, description = "Whether to gzip an object before caching it.")
    public static boolean gzipCache = false;

    // Options for training.
    @Opt(hasArg=true, description="The type of trainer to use (e.g. conditional log-likelihood, ERMA).")
    public static Trainer trainer = Trainer.CLL;

    // Options for training with ERMA.
    // TODO: Remove the "dp" prefixes on these flags.
    @Opt(hasArg=true, description="The start temperature for the softmax MBR decoder for dependency parsing.")
    public static double dpStartTemp = 10;
    @Opt(hasArg=true, description="The end temperature for the softmax MBR decoder for dependency parsing.")
    public static double dpEndTemp = .1;
    @Opt(hasArg=true, description="Whether to use log scale for the temperature annealing.")
    public static boolean dpUseLogScale = true;
    @Opt(hasArg=true, description="Whether to transition from L2DIST to the softmax MBR decoder with expected recall.")
    public static boolean dpAnnealMse = true;
    @Opt(hasArg=true, description="Whether to transition from L2DIST to the softmax MBR decoder with expected recall.")
    public static ErmaLoss dpLoss = ErmaLoss.L2DIST;

    // Options for evaluation.
    @Opt(hasArg=true, description="Whether to skip punctuation in dependency parse evaluation.")
    public static boolean dpSkipPunctuation = false;

    private static ArgParser parser;

    public JointNlpRunner() { }

    public void run() throws IOException {
        Timer t = new Timer();
        t.start();
        FastMath.useLogAddTable = useLogAddTable;
        if (useLogAddTable) {
            log.warn("Using log-add table instead of exact computation. When using global factors, this may result in numerical instability.");
        }

        // Initialize the data reader/writer.
        CorpusHandler corpus = new CorpusHandler();

        // Get a model.
        if (modelIn == null && !corpus.hasTrain()) {
        	throw new IllegalStateException("Either --modelIn or --train must be specified.");
        }

        // The annotation pipeline.
        AnnoPipeline anno = new AnnoPipeline();
        // The evaluation pipeline.
        EvalPipeline eval = new EvalPipeline();
        // The pre-processing pipeline for gold data.
        AnnoPipeline prep = new AnnoPipeline();
        JointNlpAnnotatorPrm prm = getJointNlpAnnotatorPrm();
        // Optional annotators.
        JointNlpAnnotator jointAnno = null;
        Embeddings embeds  = null;
        {
            // Pre-processing.
            RelationMunger relMunger = new RelationMunger(parser.getInstanceFromParsedArgs(RelationMungerPrm.class));
            if (CorpusHandler.getPredAts().contains(AT.REL_LABELS)) {
                RelationDataPreproc dataPreproc = relMunger.getDataPreproc();
                anno.add(dataPreproc);
                prep.add(dataPreproc);
            }
            // Annotation pipeline.
            anno.add(new EnsureStaticOptionsAreSet());
            Annotator pa = new PrefixAnnotator();
            anno.add(pa);
            prep.add(pa);
            Annotator spta = new StrictPosTagAnnotator();
            anno.add(spta);
            prep.add(spta);
            // Add Brown clusters.
            if (brownClusters != null) {
                anno.add(new BrownClusterTagger(getBrownCluterTaggerPrm(), brownClusters));
            } else {
                log.debug("No Brown clusters file specified.");
            }
            // Apply a tag map to reduce the POS tagset.
            if (reduceTags != null) {
                log.info("Reducing tags with file map: " + reduceTags);
                anno.add(new FileMapTagReducer(reduceTags));
            }
            // Add word embeddings.
            if (embeddingsFile != null) {
                Set<String> words = corpus.getAllKnownWords();
                EmbeddingsAnnotator embedAnno = new EmbeddingsAnnotator(getEmbeddingsAnnotatorPrm(), words);
                if (parser.getInstanceFromParsedArgs(RelationsFactorGraphBuilderPrm.class).useEmbeddingFeatures == true) {
                    embeds = embedAnno.getEmbeddings();
                }
                anno.add(embedAnno);
            } else {
                log.debug("No embeddings file specified.");
            }

            if (JointNlpRunner.modelIn == null) {
                // Feature selection at train time only for SRL.
                anno.add(new TransientAnnotator(new SrlFeatureSelection(prm.buPrm.fgPrm)));
            }

            if (pruneByDist) {
                // Prune via the distance-based pruner.
                anno.add(new PosTagDistancePruner());
            }
            if (pruneByModel) {
                if (pruneModel == null) {
                    throw new IllegalStateException("If pruneByModel is true, pruneModel must be specified.");
                }
                anno.add(new FirstOrderPruner(pruneModel, getJointNlpFgExampleBuilderPrm(), getDecoderPrm()));
            }
            if ((pruneByDist || pruneByModel ) && trainer == Trainer.CLL) {
                anno.add(new GoldDepParseUnpruner());
            }
            if (modelIn == null && prm.buPrm.fgPrm.srlPrm.predictPredPos) {
                // Predict SRL predicate positions as a separate step.
                // (Use the same features as the main jointAnno. These might be edited by feature selection.)
                JointNlpAnnotatorPrm prm2 = Prm.clone(prm);
                // This is transient so we need to create another one.
                prm2.crfPrm = getCrfTrainerPrm();
                // Don't include anything except for SRL.
                prm2.buPrm.fgPrm.includeDp = false;
                prm2.buPrm.fgPrm.includeRel = false;
                SrlFactorGraphBuilderPrm srlPrm = prm2.buPrm.fgPrm.srlPrm;
                srlPrm.predictPredPos = true;
                srlPrm.predictSense = false;
                srlPrm.roleStructure = RoleStructure.NO_ROLES;
                anno.add(new JointNlpAnnotator(prm2, embeds));
                // Don't predict SRL predicate position in the main jointAnno below.
                prm.buPrm.fgPrm.srlPrm.predictPredPos = false;
                prm.buPrm.fgPrm.srlPrm.roleStructure = RoleStructure.PREDS_GIVEN;
            }
            if (jointModel) {
                // Various NLP annotations.
                jointAnno = new JointNlpAnnotator(prm, embeds);
                if (modelIn != null) {
                    jointAnno.loadModel(modelIn);
                }
                anno.add(jointAnno);
            }
            // Post-processing.
            if (CorpusHandler.getPredAts().contains(AT.REL_LABELS) && !relMunger.getPrm().makeRelSingletons) {
                RelationDataPostproc dataPostproc = relMunger.getDataPostproc();
                anno.add(dataPostproc);
                prep.add(dataPostproc);
            }
        }
        {
            // Evaluation pipeline.
            if (pruneByDist || pruneByModel) {
                eval.add(new PruningEfficiency(dpSkipPunctuation));
                eval.add(new OraclePruningAccuracy(dpSkipPunctuation));
                eval.add(new OraclePruningExactMatch(dpSkipPunctuation));
            }
            if (CorpusHandler.getPredLatAts().contains(AT.POS)) {
                eval.add(new PosTagAccuracy());
            }
            if (CorpusHandler.getPredLatAts().contains(AT.DEP_TREE)) {
                eval.add(new DepParseAccuracy(dpSkipPunctuation));
                eval.add(new DepParseExactMatch(dpSkipPunctuation));
            }
            if (CorpusHandler.getPredLatAts().contains(AT.SRL_PRED_IDX)) {
                // Evaluate F1 of unlabled predicate position identification.
                eval.add(new SrlEvaluator(new SrlEvaluatorPrm(false, false, predictPredPos, false)));
                eval.add(new SrlPredIdAccuracy());
            }
            if (CorpusHandler.getPredLatAts().contains(AT.SRL)) {
                eval.add(new SrlSelfLoops());
                eval.add(new SrlEvaluator(new SrlEvaluatorPrm(true, predictSense, predictPredPos, (roleStructure != RoleStructure.NO_ROLES))));
            }
            if (CorpusHandler.getGoldOnlyAts().contains(AT.SPRL)) {
                Set<SprlClassLabel> nils = SprlClassLabel.getNils();
                if (breakdownSprlEval) {
                    for (Property q : Property.values()) {
                        //eval.add(new SprlRMSEEvaluator(roleStructure, allowPredArgSelfLoops, true, q));
                        //eval.add(new SprlRMSEEvaluator(roleStructure, allowPredArgSelfLoops, false, q));
                        eval.add(new SprlEvaluator(roleStructure, allowPredArgSelfLoops, nils, q));
                    }
                }
                //eval.add(new SprlRMSEEvaluator(roleStructure, allowPredArgSelfLoops, true));
                //eval.add(new SprlRMSEEvaluator(roleStructure, allowPredArgSelfLoops, false));
                eval.add(new SprlEvaluator(roleStructure, allowPredArgSelfLoops, nils));
            }
            if (CorpusHandler.getGoldOnlyAts().contains(AT.REL_LABELS)) {
                eval.add(new RelationEvaluator());
            }
            eval.add(new ProportionAnnotated(CorpusHandler.getPredAts()));
        }

        Experiments.trainAnnoEvalPrepGold(corpus, anno, eval, prep);

        if (corpus.hasTrain()) {
            // Save the joint model.
            if (jointAnno != null && modelOut != null) {
                jointAnno.saveModel(modelOut);
            }
            if (jointAnno != null && printModel != null) {
                jointAnno.printModel(printModel);
            }
            // Save the entire pipeline.
            if (pipeOut != null) {
                log.info("Serializing pipeline to file: " + pipeOut);
                QFiles.serialize(anno, pipeOut);
            }
        }
        t.stop();
        rep.report("elapsedSec", t.totSec());
    }

    /**
     * TODO: Deprecate this class. This is only a hold over until we remove the dependence of
     * CommunicationsAnnotator on these options being correctly set.
     *
     * @author mgormley
     */
    public static class EnsureStaticOptionsAreSet implements Annotator {
        private static final long serialVersionUID = 1L;
        private static final Logger log = LoggerFactory.getLogger(EnsureStaticOptionsAreSet.class);
        private final boolean singleRoot = InsideOutsideDepParse.singleRoot;
        private final boolean useLogAddTable = JointNlpRunner.useLogAddTable;
        @Override
        public void annotate(AnnoSentenceCollection sents) {
            log.info("Ensuring that static options (singleRoot and useLogAddTable) are set to their train-time values.");
            InsideOutsideDepParse.singleRoot = singleRoot;
            FastMath.useLogAddTable = useLogAddTable;
        }
        @Override
        public Set<AT> getAnnoTypes() {
            return Collections.emptySet();
        }
    }

    /* --------- Factory Methods ---------- */

    public static IGFeatureTemplateSelectorPrm getInformationGainFeatureSelectorPrm() {
        IGFeatureTemplateSelectorPrm prm = new IGFeatureTemplateSelectorPrm();
        prm.featureHashMod = featureHashMod;
        prm.numThreads = threads;
        prm.numToSelect = numFeatsToSelect;
        prm.maxNumSentences = numSentsForFeatSelect;
        prm.selectSense = predictSense;
        return prm;
    }

    private static JointNlpAnnotatorPrm getJointNlpAnnotatorPrm() {
        JointNlpAnnotatorPrm prm = new JointNlpAnnotatorPrm();
        prm.crfPrm = getCrfTrainerPrm();
        prm.csPrm = getCorpusStatisticsPrm();
        prm.dePrm = getDecoderPrm();
        prm.initParams = initParams;
        prm.ofcPrm = getObsFeatureConjoinerPrm();
        prm.dpSkipPunctuation = dpSkipPunctuation;
        prm.buPrm = getJointNlpFgExampleBuilderPrm();
        prm.favorSprlValidation = favorSprlValidation;
        prm.favorSrlValidation = favorSrlValidation;
        return prm;
    }

    private static JointNlpFgExampleBuilderPrm getJointNlpFgExampleBuilderPrm() {
        JointNlpFgExampleBuilderPrm prm = new JointNlpFgExampleBuilderPrm();
        // Part-of-speech tagging factor graph.
        prm.fgPrm.posPrm = getPosTagFactorGraphBuilderPrm();
        // Dependency parse factor graph.
        prm.fgPrm.dpPrm = getDepParseFactorGraphBuilderPrm();
        // SRL factor graph.
        prm.fgPrm.srlPrm = getSrlFactorGraphBuilderPrm();
        // Relation factor graph.
        if (CorpusHandler.getPredLatAts().contains(AT.REL_LABELS)) {
            prm.fgPrm.relPrm = parser.getInstanceFromParsedArgs(RelationsFactorGraphBuilderPrm.class);
        }

        boolean includeSprl = CorpusHandler.getPredLatAts().contains(AT.SPRL);
        boolean includeSrl = CorpusHandler.getPredLatAts().contains(AT.SRL);
        boolean includeIsArgVars = includeSprl && enforceSprlNilAgreement && !(sprlSrlFactors && includeSrl);

        prm.fgPrm.includePos = CorpusHandler.getPredLatAts().contains(AT.POS);
        prm.fgPrm.includeDp = CorpusHandler.getPredLatAts().contains(AT.DEP_TREE);
        prm.fgPrm.includeRel = CorpusHandler.getPredLatAts().contains(AT.REL_LABELS);
        prm.fgPrm.includeSrl = includeSrl;
        prm.fgPrm.includeSprl = includeSprl;

        // Joint features.
        if (acl14DepFeats) {
            prm.fgPrm.useSrlFeatsForLinkRoleFactors = true;
        } else {
            prm.fgPrm.useSrlFeatsForLinkRoleFactors = false;
        }

        prm.fgPrm.enforceSprlNilAgreement = enforceSprlNilAgreement;
        //prm.fgPrm.sprlPrm.sprlPipelineIndex = sprlPipelineIndex;
        prm.fgPrm.sprlSrlFactors = sprlSrlFactors;
        prm.fgPrm.sprlPrm.pairwiseFactors = sprlAllPairs;
        prm.fgPrm.sprlPrm.extraVariablesForNilAgreement = includeIsArgVars;  
        prm.fgPrm.sprlPrm.roleStructure = roleStructure;
        prm.fgPrm.sprlPrm.allowPredArgSelfLoops= allowPredArgSelfLoops;

        // TODO: probably should decouple the sprl feature extraction from the srl feature extraction
        prm.fgPrm.sprlPrm.srlFePrm = prm.fgPrm.srlPrm.srlFePrm;
        // Example construction and storage.
        prm.exPrm.cacheType = cacheType;
        prm.exPrm.gzipped = gzipCache;
        prm.exPrm.maxEntriesInMemory = maxEntriesInMemory;

        return prm;
    }

    private static PosTagFactorGraphBuilderPrm getPosTagFactorGraphBuilderPrm() {
        PosTagFactorGraphBuilderPrm posPrm = new PosTagFactorGraphBuilderPrm();
        posPrm.featureHashMod = featureHashMod;
        posPrm.posTagVarType = posTagVarType;
        return posPrm;
    }

    private static DepParseFactorGraphBuilderPrm getDepParseFactorGraphBuilderPrm() {
        DepParseFactorGraphBuilderPrm dpPrm = new DepParseFactorGraphBuilderPrm();

        // Dependency Parsing factor graph structure.
        dpPrm.linkVarType = linkVarType;
        dpPrm.useProjDepTreeFactor = useProjDepTreeFactor;
        dpPrm.unaryFactors = dpUnaryFactors;
        dpPrm.excludeNonprojectiveGrandparents = excludeNonprojectiveGrandparents;
        dpPrm.grandparentFactors = grandparentFactors;
        dpPrm.arbitrarySiblingFactors = arbitrarySiblingFactors;
        dpPrm.headBigramFactors = headBigramFactors;
        dpPrm.pruneEdges = pruneByDist || pruneByModel;

        // Dependency parsing Feature Extraction
        DepParseFeatureExtractorPrm dpFePrm = new DepParseFeatureExtractorPrm();
        dpFePrm.biasOnly = biasOnly;
        dpFePrm.firstOrderTpls = getFeatTpls(dp1FeatTpls);
        dpFePrm.secondOrderTpls = getFeatTpls(dp2FeatTpls);
        dpFePrm.featureHashMod = featureHashMod;
        dpFePrm.onlyFast = dpFastFeats;
        if (CorpusHandler.getPredLatAts().contains(AT.SRL) && acl14DepFeats) {
            // This special case is only for historical consistency.
            dpFePrm.onlyTrueBias = false;
            dpFePrm.onlyTrueEdges = false;
            dpFePrm.onlyFast = false; // Overrides command line option.
        }
        // Bitshift feature extraction.
        BitshiftDepParseFeatureExtractorPrm bsDpFePrm = parser.getInstanceFromParsedArgs(BitshiftDepParseFeatureExtractorPrm.class);
        bsDpFePrm.featureHashMod = featureHashMod;

        dpPrm.dpFePrm = dpFePrm;
        dpPrm.bsDpFePrm = bsDpFePrm;
        return dpPrm;
    }

    private static SrlFactorGraphBuilderPrm getSrlFactorGraphBuilderPrm() {
        SrlFactorGraphBuilderPrm srlPrm = new SrlFactorGraphBuilderPrm();
        // Semantic Role Labeling factor graph structure.
        srlPrm.srlVarType = srlVarType;
        srlPrm.makeUnknownPredRolesLatent = makeUnknownPredRolesLatent;
        srlPrm.roleStructure = roleStructure;
        srlPrm.allowPredArgSelfLoops = allowPredArgSelfLoops;
        srlPrm.unaryFactors = srlUnaryFactors;
        srlPrm.binarySenseRoleFactors = binarySenseRoleFactors;
        srlPrm.predictSense = predictSense;
        srlPrm.predictPredPos = predictPredPos;
        srlPrm.fcmFactors = srlFcmFactors;
        srlPrm.fcmFineTuning = srlFcmFineTuning;
        srlPrm.fcmWfPrm = parser.getInstanceFromParsedArgs(SrlWordFeaturesPrm.class);

        // SRL Feature Extraction.
        SrlFeatureExtractorPrm srlFePrm = new SrlFeatureExtractorPrm();
        srlFePrm.biasOnly = biasOnly;
        srlFePrm.useTemplates = useTemplates;
        srlFePrm.senseTemplates = getFeatTpls(senseFeatTpls);
        srlFePrm.argTemplates = getFeatTpls(argFeatTpls);
        srlFePrm.featureHashMod = featureHashMod;
        if (noLemma) {
            srlFePrm.argTemplates = TemplateLanguage.filterOutFeats(srlFePrm.argTemplates, TokProperty.LEMMA);
            srlFePrm.senseTemplates = TemplateLanguage.filterOutFeats(srlFePrm.senseTemplates, TokProperty.LEMMA);
        }
        srlPrm.srlFePrm = srlFePrm;
        return srlPrm;
    }

    private static ObsFeatureConjoinerPrm getObsFeatureConjoinerPrm() {
        ObsFeatureConjoinerPrm prm = new ObsFeatureConjoinerPrm();
        prm.featCountCutoff = featCountCutoff;
        return prm;
    }

    /**
     * Gets feature templates from multiple files or resources.
     * @param featTpls A colon separated list of paths to feature template files or resources.
     * @return The feature templates from all the paths.
     */
    private static List<FeatTemplate> getFeatTpls(String featTpls) {
        Collection<FeatTemplate> tpls = new LinkedHashSet<FeatTemplate>();

        TemplateReader tr = new TemplateReader();
        for (String path : featTpls.split(":")) {
            if (path.equals("coarse1") || path.equals("coarse2")) {
                List<FeatTemplate> coarseUnigramSet;
                if (path.equals("coarse1")) {
                    coarseUnigramSet = TemplateSets.getCoarseUnigramSet1();
                } else if (path.equals("coarse2")) {
                    coarseUnigramSet = TemplateSets.getCoarseUnigramSet2();
                } else {
                    throw new IllegalStateException();
                }
                tpls.addAll(coarseUnigramSet);
            } else {
                try {
                    tr.readFromFile(path);
                } catch (IOException e) {
                    try {
                        tr.readFromResource(path);
                    } catch (IOException e1) {
                        throw new IllegalStateException("Unable to read templates as file or resource: " + path, e1);
                    }
                }
            }
        }
        tpls.addAll(tr.getTemplates());

        return new ArrayList<FeatTemplate>(tpls);
    }

    public static CorpusStatisticsPrm getCorpusStatisticsPrm() {
        CorpusStatisticsPrm prm = new CorpusStatisticsPrm();
        prm.cutoff = cutoff;
        prm.language = CorpusHandler.language;
        prm.useGoldSyntax = CorpusHandler.useGoldSyntax;
        return prm;
    }

    private static CrfTrainerPrm getCrfTrainerPrm() {
        FgInferencerFactory infPrm = getInfFactory();

        CrfTrainerPrm prm = new CrfTrainerPrm();
        prm.infFactory = infPrm;
        if (infPrm instanceof BeliefsModuleFactory) {
            // TODO: This cast is a temporary hack.
            prm.bFactory = (BeliefsModuleFactory) infPrm;
        }
        Pair<Optimizer<DifferentiableFunction>, Optimizer<DifferentiableBatchFunction>> opts = OptimizerFactory.getOptimizers();
        prm.optimizer = opts.get1();
        prm.batchOptimizer = opts.get2();
        prm.trainer = trainer;

        // TODO: add options for other loss functions.
        if (prm.trainer == Trainer.ERMA) {
            if (dpLoss == ErmaLoss.SOFTMAX_MBR) {
                if (!CorpusHandler.getPredAts().equals(QSets.getSet(AT.DEP_TREE))) {
                    throw new RuntimeException("The " + dpLoss.name() + " loss function is only for " + AT.DEP_TREE);
                }
                DepParseSoftmaxMbrFactory lossPrm = new DepParseSoftmaxMbrFactory();
                lossPrm.annealMse = dpAnnealMse;
                lossPrm.startTemp = dpStartTemp;
                lossPrm.useLogScale = dpUseLogScale;
                lossPrm.endTemp = dpEndTemp;
                prm.dlFactory = lossPrm;
            } else if (dpLoss == ErmaLoss.L2DIST) {
                prm.dlFactory = new L2DistanceFactory();
            } else if (dpLoss == ErmaLoss.EXPECTED_RECALL) {
                prm.dlFactory = new ExpectedRecallFactory();
            } else {
                throw new RuntimeException("Unsupported loss: " + dpLoss.name());
            }
        }

        return prm;
    }

    private static FgInferencerFactory getInfFactory() {
        if (inference == Inference.BRUTE_FORCE) {
            BruteForceInferencerPrm prm = new BruteForceInferencerPrm(algebra.getAlgebra());
            return prm;
        } else if (inference == Inference.BP) {
            BeliefPropagationPrm bpPrm = new BeliefPropagationPrm();
            bpPrm.s = algebra.getAlgebra();
            bpPrm.schedule = bpSchedule;
            bpPrm.updateOrder = bpUpdateOrder;
            bpPrm.normalizeMessages = normalizeMessages;
            bpPrm.maxIterations = bpMaxIterations;
            bpPrm.convergenceThreshold = bpConvergenceThreshold;
            bpPrm.keepTape = (trainer == Trainer.ERMA);
            if (bpDumpDir != null) {
                bpPrm.dumpDir = Paths.get(bpDumpDir.getAbsolutePath());
            }
            return bpPrm;
        } else if (inference == Inference.DP) {
            if (CorpusHandler.getPredAts().equals(QSets.getSet(AT.DEP_TREE))
                    && grandparentFactors && !arbitrarySiblingFactors && !headBigramFactors) {
                return new O2AllGraFgInferencerFactory(algebra.getAlgebra());
            } else {
                throw new IllegalStateException("DP inference only supported for dependency parsing with all grandparent factors.");
            }
        } else {
            throw new RuntimeException("Unsupported inference method: " + inference);
        }
    }

    private static JointNlpDecoderPrm getDecoderPrm() {
        MbrDecoderPrm mbrPrm = new MbrDecoderPrm();
        mbrPrm.infFactory = getInfFactory();
        mbrPrm.loss = Loss.L1;
        JointNlpDecoderPrm prm = new JointNlpDecoderPrm();
        prm.mbrPrm = mbrPrm;
        return prm;
    }

    private static BrownClusterTaggerPrm getBrownCluterTaggerPrm() {
        BrownClusterTaggerPrm bcPrm = new BrownClusterTaggerPrm();
        bcPrm.language = CorpusHandler.language;
        bcPrm.maxTagLength = bcMaxTagLength;
        return bcPrm;
    }

    private static EmbeddingsAnnotatorPrm getEmbeddingsAnnotatorPrm() {
        EmbeddingsAnnotatorPrm prm = new EmbeddingsAnnotatorPrm();
        prm.embeddingsFile = embeddingsFile;
        prm.embNorm = embNorm;
        prm.embScalar= embScalar;
        prm.entitySpecificEmbeddings = entitySpecificEmbeddings;
        return prm;
    }

    public static void main(String[] args) throws IOException {
        ArgParser parser = new ArgParser(JointNlpRunner.class);
        parser.registerClass(JointNlpRunner.class);
        parser.registerClass(OptimizerFactory.class);
        parser.registerClass(CorpusHandler.class);
        parser.registerClass(RelationMungerPrm.class);
        parser.registerClass(RelationsFactorGraphBuilderPrm.class);
        parser.registerClass(InsideOutsideDepParse.class);
        parser.registerClass(ReporterManager.class);
        parser.registerClass(BitshiftDepParseFeatureExtractorPrm.class);
        parser.registerClass(SrlWordFeaturesPrm.class);
        parser.parseArgs(args);
        JointNlpRunner.parser = parser;

        ReporterManager.init(ReporterManager.reportOut, true);
        Prng.seed(seed);
        Threads.initDefaultPool(threads);

        JointNlpRunner pipeline = new JointNlpRunner();
        pipeline.run();
    }

    
}
