package edu.jhu.nlp.data;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import edu.jhu.prim.tuple.Pair;
import org.apache.commons.lang3.ObjectUtils;

/**
 * A relation mention consisting of:
 * <ul>
 * <li>Label of the relation type.</li>
 * <li>An optional text span corresponding to the trigger.</li>
 * <li>A list of arguments, each of which has a text span and a label.</li>
 * </ul>
 * 
 * @author mgormley
 */
public class RelationMention {

    private String type;
    private String subType;
    private List<Pair<String, NerMention>> args;
    // Optional trigger span for this situation.
    private Span trigger;

    public RelationMention(String type, String subType, List<Pair<String, NerMention>> args, Span trigger) {
        super();
        this.type = type;
        this.subType = subType;
        this.args = args;
        this.trigger = trigger;
    }

    public RelationMention(RelationMention other) {
        this.type = other.type;
        this.subType = other.subType;
        if (args != null) {
            this.args = new ArrayList<>(other.args.size());
            for (Pair<String, NerMention> pair : other.args) {
                this.args.add(new Pair<String, NerMention>(pair.get1(), new NerMention(pair.get2())));
            }
        }
        this.trigger = other.trigger;
    }
    
    public String getType() {
        return type;
    }

    public String getSubType() {
        return subType;
    }

    public List<Pair<String, NerMention>> getArgs() {
        return args;
    }

    public List<Pair<String, NerMention>> getNerOrderedArgs() {
        List<Pair<String, NerMention>> argsOrd = new ArrayList<>(args);
        Collections.sort(argsOrd, new Comparator<Pair<String,NerMention>>(){
            @Override
            public int compare(Pair<String, NerMention> o1, Pair<String, NerMention> o2) {
                int diff = ObjectUtils.compare(o1.get2(), o2.get2());
                if (diff != 0) { return diff; }
                return ObjectUtils.compare(o1.get1(), o2.get1());
            }
        });
        return argsOrd;
    }

    public Span getTrigger() {
        return trigger;
    }

    public void intern() {
        if (type != null) {
            type = type.intern();
        }
        if (subType != null) {
            subType = subType.intern();
        }
        for (int i = 0; i < args.size(); i++) {
            Pair<String, NerMention> pair = args.get(i);
            pair.get2().intern();
            pair = new Pair<>(pair.get1().intern(), pair.get2());
            args.set(i, pair);
        }
    }

    @Override
    public String toString() {
        return "SituationMent [type=" + type + ", subType=" + subType + ", args=" + args + ", trigger=" + trigger + "]";
    }

    public String toString(List<String> words) {
        StringBuilder argsStr = new StringBuilder();
        argsStr.append("[");
        for (int i = 0; i < args.size(); i++) {
            if (i != 0) {
                argsStr.append(", ");
            }
            Pair<String, NerMention> p = args.get(i);
            argsStr.append(p.get1());
            argsStr.append("=");
            argsStr.append(p.get2().getSpan().getString(words, " "));
        }
        argsStr.append("]");
        String triggerStr = null;
        if (trigger != null) {
            triggerStr = trigger.getString(words, " ");
        }
        return "SituationMent [type=" + type + ", subType=" + subType + ", args=" + argsStr + ", trigger="
                + triggerStr + "]";
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + ((args == null) ? 0 : getNerOrderedArgs().hashCode());
        result = prime * result + ((subType == null) ? 0 : subType.hashCode());
        result = prime * result + ((trigger == null) ? 0 : trigger.hashCode());
        result = prime * result + ((type == null) ? 0 : type.hashCode());
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (getClass() != obj.getClass())
            return false;
        RelationMention other = (RelationMention) obj;
        if (args == null) {
            if (other.args != null)
                return false;
        } else if (!getNerOrderedArgs().equals(other.getNerOrderedArgs()))
            return false;
        if (subType == null) {
            if (other.subType != null)
                return false;
        } else if (!subType.equals(other.subType))
            return false;
        if (trigger == null) {
            if (other.trigger != null)
                return false;
        } else if (!trigger.equals(other.trigger))
            return false;
        if (type == null) {
            if (other.type != null)
                return false;
        } else if (!type.equals(other.type))
            return false;
        return true;
    }
    
    

}