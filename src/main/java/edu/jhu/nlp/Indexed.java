package edu.jhu.nlp;

import java.util.Iterator;

/**
 * An object paired with an integer to facilitate enumerate
 *
 * @param <T>
 */
public class Indexed<T> {

    private int index;
    private T obj;
    public Indexed(T obj, int index) {
        this.obj = obj;
        this.index = index;
    }

    public int index() {
        return index;
    }

    public T get() {
        return obj;
    }

    public String toString() {
        return String.format("(%s, %d)", get(), index());
    }
    
    public static <T> Iterable<Indexed<T>> enumerate(Iterable<T> stream) {
        return new Iterable<Indexed<T>>() {
            
            @Override
            public Iterator<Indexed<T>> iterator() {
                Iterator<T> itr = stream.iterator();
                return new Iterator<Indexed<T>>() {
                    
                    private int i = 0;
                    
                    @Override
                    public boolean hasNext() {
                        return itr.hasNext();
                    }
                    
                    @Override
                    public Indexed<T> next() {
                        Indexed<T> nextPair = new Indexed<T>(itr.next(), i);
                        i++;
                        return nextPair;
                    }
                };
            }
        };
    }

}
