package edu.jhu.nlp.embed;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.io.Serializable;
import java.util.regex.Pattern;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.pacaya.autodiff.Tensor;
import edu.jhu.pacaya.util.semiring.RealAlgebra;
import edu.jhu.prim.Primitives.MutableInt;
import edu.jhu.prim.arrays.DoubleArrays;
import edu.jhu.prim.bimap.IntObjectBimap;

/**
 * Storage for a set of word embeddings. Also contains a method to load embeddings from a text file. 
 * @author mgormley
 */
public class Embeddings implements Serializable {
    
    public enum Scaling { NONE, L1_NORM, L2_NORM, STD_NORMAL }
    
    private static final long serialVersionUID = 1L;
    private static final Logger log = LoggerFactory.getLogger(Embeddings.class);
    private static final Pattern DIGITS = Pattern.compile("[0-9]");
    private static final String UNKNOWN_WORD = "<UNK>";
    private final Tensor embeds;
    private final IntObjectBimap<String> alphabet;
    
    public Embeddings(Tensor embeds, IntObjectBimap<String> alphabet) {
        this.embeds = embeds;
        this.alphabet = alphabet;
    }
    
    public Embeddings(File txtFile) {
        log.info("Reading word embeddings from file: " + txtFile);
        // Count the number of words and length of the embeddings.
        final MutableInt numWords = new MutableInt(0);
        final MutableInt dim = new MutableInt(-1);
        EmbeddingHandler countHandler = new EmbeddingHandler(){
            @Override
            public void addEmbedding(String word, double[] embed) {
                numWords.v += 1;
                dim.v = embed.length;
            }
        };
        parseEmbFile(txtFile, countHandler);
        // Store the embeddings.
        embeds = new Tensor(RealAlgebra.getInstance(), numWords.v, dim.v);
        alphabet = new IntObjectBimap<String>();
        EmbeddingHandler addHandler = new EmbeddingHandler(){
            @Override
            public void addEmbedding(String word, double[] embed) {
                int i = alphabet.lookupIndex(word);
                for (int d=0; d<embed.length; d++) {
                    embeds.set(embed[d], i, d);
                }
            }
        };
        parseEmbFile(txtFile, addHandler);
        // TODO: Always add an embedding for the special <UNK> word type.
        // Currently this is commented out to align with Mo's implementation.
        //        if (alphabet.lookupIndex(UNKNOWN_WORD) == -1) {
        //            addHandler.addEmbedding(UNKNOWN_WORD, new double[dim.v]);
        //        }
        alphabet.stopGrowth();
        log.debug("Embedding vocabulary size: " + alphabet.size());
    }
    
    private interface EmbeddingHandler {
        void addEmbedding(String word, double[] embed);
    }

    /**
     * Loads the embeddings from a tab-separated text file in UTF-8. Each line consists of n+1 columns for an
     * n-dimensional embedding. The first column is the word. The i+1st column is the ith dimension
     * of the embedding for that word.
     * 
     * @param txtFile The UTF-8 encoded tsv file.
     * @throws IOException
     */
    public static void parseEmbFile(File txtFile, EmbeddingHandler handler) {
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(txtFile), "UTF-8"))) {
            String line;
            Pattern tab = Pattern.compile("\t");
            int dim = -1;
            while ((line = reader.readLine()) != null) {
                String[] splits = tab.split(line);
                if (splits.length > 0) {
                    String word = splits[0];
                    double[] embed = new double[splits.length - 1];
                    for (int i=1; i<splits.length; i++) {
                        embed[i-1] = Double.parseDouble(splits[i]);
                    }
                    handler.addEmbedding(word, embed);
                    if (dim != -1 && embed.length != dim) {
                        throw new RuntimeException("Read dimension with incorrect length: " + word + " " + dim);
                    }
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
        
    /** Does a "smart" lookup of the embedding by trying various forms of the word as alternatives. */
    // TODO: Could consider frequency of word when selecting embedding.
    public int findEmbedding(String word) {
        int i = -1;
        if ((i = alphabet.lookupIndex(word)) == -1) {
            word = DIGITS.matcher(word).replaceAll("#");
            if ((i = alphabet.lookupIndex(word)) == -1) {
                word = word.toLowerCase();
                if ((i = alphabet.lookupIndex(word)) == -1) {
                    word = word.toUpperCase();
                    if ((i = alphabet.lookupIndex(word)) == -1) {
                        word = UNKNOWN_WORD;
                    }
                }
            }
        }
        return i;
    }
    
    public Tensor getEmbeddings() {
        return embeds;
    }

    public IntObjectBimap<String> getAlphabet() {
        return alphabet;
    }

    private void writeEmbeddings(PrintStream out) {
        for (int i=0; i<embeds.getDim(0); i++) {
            String word = alphabet.lookupObject(i);
            out.print(word);
            for (int d=0; d<embeds.getDim(1); d++) {
                out.print("\t");
                out.print(embeds.get(i, d));
            }
            out.println();
        }
    }
    
    public void normPerWord(Scaling sc) {
        if (sc == Scaling.NONE || sc == null){
            return;
        }
        for (int i=0; i<embeds.getDim(0); i++) {
            // Get the embedding.
            double[] emb = new double[embeds.getDim(1)];
            for (int d=0; d<emb.length; d++) {
                emb[d] = embeds.get(i,d);
            }
            // Normalize.
            if (sc == Scaling.L1_NORM) {
                DoubleArrays.scale(emb, 1.0 / DoubleArrays.l1norm(emb));
            } else if (sc == Scaling.L2_NORM) {
                DoubleArrays.scale(emb, 1.0 / DoubleArrays.l2norm(emb));
            } else if (sc == Scaling.STD_NORMAL) {
                DoubleArrays.add(emb, -DoubleArrays.mean(emb));
                DoubleArrays.scale(emb, 1.0 / DoubleArrays.stdDev(emb));
            } else {
                throw new RuntimeException("Unsupported method rescaling: " + sc);
            }
            // Set the embedding.
            for (int d=0; d<emb.length; d++) {
                embeds.set(emb[d], i, d);
            }
        }
    }

    public void scaleAll(double alpha) {
        embeds.multiply(alpha);
    }
    
    public static void main(String[] args) throws IOException {
        Embeddings embed = new Embeddings(new File(args[0]));
        embed.writeEmbeddings(System.out);
    }
    
}
