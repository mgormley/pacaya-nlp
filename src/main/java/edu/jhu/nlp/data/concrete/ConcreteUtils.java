package edu.jhu.nlp.data.concrete;

import java.util.ArrayList;
import java.util.List;

import edu.jhu.hlt.concrete.AnnotationMetadata;
import edu.jhu.hlt.concrete.Communication;
import edu.jhu.hlt.concrete.DependencyParse;
import edu.jhu.hlt.concrete.EntityMentionSet;
import edu.jhu.hlt.concrete.Parse;
import edu.jhu.hlt.concrete.Section;
import edu.jhu.hlt.concrete.Sentence;
import edu.jhu.hlt.concrete.SituationMentionSet;
import edu.jhu.hlt.concrete.TextSpan;
import edu.jhu.hlt.concrete.Token;
import edu.jhu.hlt.concrete.TokenList;
import edu.jhu.hlt.concrete.TokenTagging;
import edu.jhu.hlt.concrete.Tokenization;
import edu.jhu.hlt.concrete.TokenizationKind;
import edu.jhu.hlt.concrete.communications.CommunicationFactory;
import edu.jhu.hlt.concrete.section.SectionFactory;
import edu.jhu.hlt.concrete.util.ConcreteException;
import edu.jhu.hlt.concrete.uuid.UUIDFactory;
import edu.jhu.nlp.data.simple.AnnoSentence;
import edu.jhu.nlp.data.simple.AnnoSentenceCollection;

public class ConcreteUtils {

    private ConcreteUtils() { }

    /**
     * @return A concrete communication containing the tokens of the text, split into sections by newlines (skipping blank lines), and tokenized on whitespace which are replaced by single spaces
     * @throws ConcreteException
     */
    public static Communication ingestText(String text, String commId, String commTool, String tokTool) {
        AnnotationMetadata commMetadata = new AnnotationMetadata();
        commMetadata.setTimestamp(System.currentTimeMillis());
        commMetadata.setTool(commTool);
        AnnotationMetadata tokenizationMetadata = new AnnotationMetadata();
        tokenizationMetadata.setTimestamp(System.currentTimeMillis());
        tokenizationMetadata.setTool(tokTool);
        Communication comm = new Communication();
        comm.setId(commId);
        comm.setType("corpus");
        comm.setMetadata(commMetadata);

        List<Section> sections = new ArrayList<>();
        List<String> sentenceStrs = new ArrayList<>();
        int nchars = 0;
        for (String sentStr : text.trim().split("\\n")) {
            sentStr = sentStr.trim();
            if (sentStr.length() == 0) { continue; }
            List<String> tokenStrs = new ArrayList<>();
            List<Token> tokList = new ArrayList<>();
            int tokIndex = 0;
            for (String tokStr : sentStr.split("\\s+")) {
                tokStr = tokStr.trim();

                Token newTok = new Token();
                newTok.setTokenIndex(tokIndex);
                newTok.setTextSpan(new TextSpan(nchars, nchars + tokStr.length()));
                newTok.setText(tokStr);

                tokList.add(newTok);
                tokenStrs.add(tokStr);
                tokIndex += 1;
                nchars += tokStr.length() + 1;  // +1 because of space or newline
            }
            Tokenization tok = new Tokenization(UUIDFactory.newUUID(), tokenizationMetadata, TokenizationKind.TOKEN_LIST);
            Section newSection = new Section(UUIDFactory.newUUID(), "section");
            Sentence sent = new Sentence(UUIDFactory.newUUID());

            tok.setTokenList(new TokenList(tokList));
            sent.setTokenization(tok);
            newSection.addToSentenceList(sent);

            sections.add(newSection);
            sentenceStrs.add(String.join(" ", tokenStrs));
        }
        comm.setSectionList(sections);
        comm.setText(String.join("\n", sentenceStrs));
        return comm;
    }

    public static String getText(AnnoSentenceCollection sents) {
        List<String> sentences = new ArrayList<>();
        for (AnnoSentence s : sents) {
            sentences.add(String.join(" ", s.getWords()));
        }
        return String.join("\n", sentences);
    }

    public static TokenTagging getFirstXTags(Tokenization tokenization, String taggingType) {
        return getFirstXTagsWithName(tokenization, taggingType, null);
    }

    public static TokenTagging getFirstXTagsWithName(Tokenization tokenization, String taggingType, String toolName) {
        if (!tokenization.isSetTokenTaggingList()) {
            return null;
        }
        List<TokenTagging> tokenTaggingLists = tokenization.getTokenTaggingList();
        for (int i = 0; i < tokenTaggingLists.size(); i++) {
            TokenTagging tt = tokenTaggingLists.get(i);
            if (tt.isSetTaggingType() && tt.getTaggingType().equals(taggingType)
                    && (toolName == null || tt.getMetadata().getTool().contains(toolName))) {
                return tt;
            }
        }
        return null;
    }

    public static DependencyParse getFirstDependencyParse(Tokenization tokenization) {
        return getFirstDependencyParseWithName(tokenization, null);
    }

    public static DependencyParse getFirstDependencyParseWithName(Tokenization tokenization, String toolName) {
        List<DependencyParse> parseList = tokenization.getDependencyParseList();
        if (parseList == null) {
            return null;
        }
        for (int i = 0; i < parseList.size(); i++) {
            DependencyParse dp = parseList.get(i);
            if (toolName == null || dp.getMetadata().getTool().contains(toolName)) {
                return dp;
            }
        }
        return null;
    }

    public static Parse getFirstParse(Tokenization tokenization) {
        return getFirstParseWithName(tokenization, null);
    }

    public static Parse getFirstParseWithName(Tokenization tokenization, String toolName) {
        List<Parse> parseList = tokenization.getParseList();
        if (parseList == null) {
            return null;
        }
        for (int i = 0; i < parseList.size(); i++) {
            Parse p = parseList.get(i);
            if (toolName == null || p.getMetadata().getTool().contains(toolName)) {
                return p;
            }
        }
        return null;
    }

    public static int getNumSents(Communication comm) {
        int n = 0;
        for (Section section : comm.getSectionList()) {
            n += section.getSentenceListSize();
        }
        return n;
    }

    public static EntityMentionSet getFirstEntityMentionSetWithName(Communication comm, String toolName) {
        List<EntityMentionSet> cEmsList = comm.getEntityMentionSetList();
        if (cEmsList == null) {
            return null;
        }
        for (EntityMentionSet cEms : cEmsList) {
            if (toolName == null || cEms.getMetadata().getTool().contains(toolName)) {
                return cEms;
            }
        }
        return null;
    }

    public static SituationMentionSet getFirstSituationMentionSetWithName(Communication comm, String toolName) {
        List<SituationMentionSet> cSmsList = comm.getSituationMentionSetList();
        if (cSmsList == null) {
            return null;
        }
        for (SituationMentionSet cSms : cSmsList) {
            if (toolName == null || cSms.getMetadata().getTool().contains(toolName)) {
                return cSms;
            }
        }
        return null;
    }

}
