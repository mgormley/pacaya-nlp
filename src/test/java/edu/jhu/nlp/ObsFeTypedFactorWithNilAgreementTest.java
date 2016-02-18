package edu.jhu.nlp;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;
import java.util.List;

import org.junit.Test;

import edu.jhu.nlp.joint.JointNlpFactorGraph.IsArgLabel;
import edu.jhu.nlp.sprl.SprlClassLabel;
import edu.jhu.nlp.sprl.SprlFactorGraphBuilder.SprlVar;
import edu.jhu.nlp.srl.SrlFactorGraphBuilder.RoleVar;
import edu.jhu.pacaya.gm.model.Var;
import edu.jhu.pacaya.gm.model.Var.VarType;
import edu.jhu.pacaya.gm.model.VarConfig;
import edu.jhu.pacaya.gm.model.VarSet;
import edu.jhu.prim.tuple.Pair;

public class ObsFeTypedFactorWithNilAgreementTest {

    @Test
    public void testIsArgSprlAllOrNoneNil() {
        // test the Binary case
        int i = 10;
        int j = 15;
        int q = 3;
        Var argVar = new Var(VarType.LATENT, 2, "isarg" + i + "_" + j, IsArgLabel.labels);
        String sprlName = "sprl_r" + i + "-" + "a" + j + "_" + q;
        SprlVar sprlVar = new SprlVar(null, SprlClassLabel.getLabels().size(), sprlName, SprlClassLabel.getLabels(), i,
                j);
        VarSet vSet = new VarSet(argVar, sprlVar);
        List<Pair<Enum<?>, Enum<?>>> goodPairs = Arrays.asList(new Pair<>(IsArgLabel.IS_ARG, SprlClassLabel.UNLIKELY),
                new Pair<>(IsArgLabel.IS_ARG, SprlClassLabel.UNKNOWN),
                new Pair<>(IsArgLabel.IS_ARG, SprlClassLabel.LIKELY),
                new Pair<>(IsArgLabel.NOT_AN_ARG, SprlClassLabel.NOT_AN_ARG));
        for (Pair<Enum<?>, Enum<?>> goodPair : goodPairs) {
            VarConfig vc = new VarConfig();
            vc.put(argVar, goodPair.get1().ordinal());
            vc.put(sprlVar, goodPair.get2().ordinal());
            assertTrue(ObsFeTypedFactorWithNilAgreement.allOrNoneNil(vSet, Arrays.asList(argVar, sprlVar),
                    Arrays.asList(IsArgLabel.NOT_AN_ARG.ordinal(), SprlClassLabel.NOT_AN_ARG.ordinal()),
                    vc.getConfigIndex()));
        }
        List<Pair<Enum<?>, Enum<?>>> badPairs = Arrays.asList(new Pair<>(IsArgLabel.IS_ARG, SprlClassLabel.NOT_AN_ARG),
                new Pair<>(IsArgLabel.NOT_AN_ARG, SprlClassLabel.UNLIKELY),
                new Pair<>(IsArgLabel.NOT_AN_ARG, SprlClassLabel.UNKNOWN),
                new Pair<>(IsArgLabel.NOT_AN_ARG, SprlClassLabel.LIKELY));
        for (Pair<Enum<?>, Enum<?>> badPair : badPairs) {
            VarConfig vc = new VarConfig();
            vc.put(argVar, badPair.get1().ordinal());
            vc.put(sprlVar, badPair.get2().ordinal());
            assertFalse(ObsFeTypedFactorWithNilAgreement.allOrNoneNil(vSet, Arrays.asList(argVar, sprlVar),
                    Arrays.asList(IsArgLabel.NOT_AN_ARG.ordinal(), SprlClassLabel.NOT_AN_ARG.ordinal()),
                    vc.getConfigIndex()));
        }

    }

    @Test
    public void testRoleSprlAllOrNoneNil() {
        // test the Binary case
        int i = 10;
        int j = 15;
        int q = 3;
        List<String> stateNames = Arrays.asList("ARG0", "ARG1", "_", "ARG2");
        RoleVar roleVar = new RoleVar(VarType.PREDICTED, stateNames.size(), "role_var", stateNames, i, j);
        assertTrue(roleVar.getNilState() == 2);
        String sprlName = "sprl_r" + i + "-" + "a" + j + "_" + q;
        SprlVar sprlVar = new SprlVar(null, SprlClassLabel.getLabels().size(), sprlName, SprlClassLabel.getLabels(), i,
                j);
        VarSet vSet = new VarSet(roleVar, sprlVar);
        List<Pair<Integer, Enum<?>>> goodPairs = Arrays.asList(new Pair<>(0, SprlClassLabel.UNKNOWN),
                new Pair<>(1, SprlClassLabel.UNLIKELY), new Pair<>(3, SprlClassLabel.LIKELY),
                new Pair<>(1, SprlClassLabel.UNKNOWN), new Pair<>(0, SprlClassLabel.UNLIKELY),
                new Pair<>(3, SprlClassLabel.LIKELY), new Pair<>(1, SprlClassLabel.UNKNOWN),
                new Pair<>(3, SprlClassLabel.UNLIKELY), new Pair<>(0, SprlClassLabel.LIKELY),
                new Pair<>(3, SprlClassLabel.UNKNOWN), new Pair<>(0, SprlClassLabel.UNLIKELY),
                new Pair<>(1, SprlClassLabel.LIKELY), new Pair<>(3, SprlClassLabel.UNKNOWN),
                new Pair<>(1, SprlClassLabel.UNLIKELY), new Pair<>(0, SprlClassLabel.LIKELY),
                new Pair<>(roleVar.getNilState(), SprlClassLabel.NOT_AN_ARG));
        for (Pair<Integer, Enum<?>> goodPair : goodPairs) {
            VarConfig vc = new VarConfig();
            vc.put(roleVar, goodPair.get1());
            vc.put(sprlVar, goodPair.get2().ordinal());
            assertTrue(ObsFeTypedFactorWithNilAgreement.allOrNoneNil(vSet, Arrays.asList(roleVar, sprlVar),
                    Arrays.asList(roleVar.getNilState(), SprlClassLabel.NOT_AN_ARG.ordinal()),
                    vc.getConfigIndex()));
        }
        List<Pair<Integer, Enum<?>>> badPairs = Arrays.asList(new Pair<>(0, SprlClassLabel.NOT_AN_ARG),
                new Pair<>(1, SprlClassLabel.NOT_AN_ARG), new Pair<>(3, SprlClassLabel.NOT_AN_ARG),
                new Pair<>(roleVar.getNilState(), SprlClassLabel.UNLIKELY),
                new Pair<>(roleVar.getNilState(), SprlClassLabel.UNKNOWN),
                new Pair<>(roleVar.getNilState(), SprlClassLabel.LIKELY));
        for (Pair<Integer, Enum<?>> badPair : badPairs) {
            VarConfig vc = new VarConfig();
            vc.put(roleVar, badPair.get1());
            vc.put(sprlVar, badPair.get2().ordinal());
            assertFalse(ObsFeTypedFactorWithNilAgreement.allOrNoneNil(vSet, Arrays.asList(roleVar, sprlVar),
                    Arrays.asList(roleVar.getNilState(), SprlClassLabel.NOT_AN_ARG.ordinal()),
                    vc.getConfigIndex()));
        }

    }

}
