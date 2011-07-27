package org.graphstream.graph.implementations;

import org.graphstream.graph.Element;
import org.graphstream.util.NativePointer;

import java.nio.Buffer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

public class NativeElement implements Element {

    static {
	System.loadLibrary("gs_native");
    }

    public static void main(String ... args) {
	NativeElement e1, e2;
	
	e1 = new NativeElement("e1");
	e2 = new NativeElement("e2");

	e1 = null;
	e2 = null;
	
	System.gc();
    }

    private final NativePointer  __ref;
    private final String id;

    @Deprecated
    private NativeElement() {
	// Do not call this constructor
	this.id = null;
	this.__ref = new NativePointer();
    }

    public NativeElement(String id) {
	this.id = id;
	this.__ref = new NativePointer();

	/*
	 * Alloc native ressources.
	 */
	init();
    }

    /**
     * WARNING : this method is protected to be overriden only by
     * other native classes extending NativeElement, such as
     * NativeNode or NativeEdge.
     * 
     * You do not have to call or use this method.
     */
    protected native void init();

    /**
     * WARNING : this method is protected to be overriden only by
     * other native classes extending NativeElement, such as
     * NativeNode or NativeEdge.
     * 
     * You do not have to call or use this method.
     */
    protected native void uninit();
    
    @Override
    protected void finalize() throws Throwable {
	super.finalize();

	/*
	 * Free native ressources.
	 */
	uninit();
    }

    /**
     * Unique identifier of this element.
     * 
     * @return The identifier value.
     */
    public String getId() {
	return id;
    }

    /**
     * Get the attribute object bound to the given key. The returned value may
     * be null to indicate the attribute does not exists or is not supported.
     * 
     * @param key
     *            Name of the attribute to search.
     * @return The object bound to the given key or null if no object match this
     *         attribute name.
     */
    public native <T> T getAttribute(String key);

    /**
     * Like {@link #getAttribute(String)}, but returns the first existing
     * attribute in a list of keys, instead of only one key. The key list order
     * matters.
     * 
     * @param keys
     *            Several strings naming attributes.
     * @return The first attribute that exists.
     */
    public native <T> T getFirstAttributeOf(String... keys);
    
    /**
     * Get the attribute object bound to the given key if it is an instance of
     * the given class. Some The returned value maybe null to indicate the
     * attribute does not exists or is not an instance of the given class.
     * 
     * @param key
     *            The attribute name to search.
     * @param clazz
     *            The expected attribute class.
     * @return The object bound to the given key or null if no object match this
     *         attribute.
     */
    public native <T> T getAttribute(String key, Class<T> clazz);
    
    /**
     * Like {@link #getAttribute(String, Class)}, but returns the first existing
     * attribute in a list of keys, instead of only one key. The key list order
     * matters.
     * 
     * @param clazz
     *            The class the attribute must be instance of.
     * @param keys
     *            Several string naming attributes.
     * @return The first attribute that exists.
     */
    public native <T> T getFirstAttributeOf(Class<T> clazz, String... keys);
    
    /**
     * Get the label string bound to the given key key. Labels are special
     * attributes whose value is a character sequence. If an attribute with the
     * same name exists but is not a character sequence, null is returned.
     * 
     * @param key
     *            The label to search.
     * @return The label string value or null if not found.
     */
    public native CharSequence getLabel(String key);
    
    /**
     * Get the number bound to key. Numbers are special attributes whose value
     * is an instance of Number. If an attribute with the same name exists but
     * is not a Number, NaN is returned.
     * 
     * @param key
     *            The name of the number to search.
     * @return The number value or NaN if not found.
     */
    public native double getNumber(String key);
    
    /**
     * Get the vector of number bound to key. Vectors of numbers are special
     * attributes whose value is a sequence of numbers. If an attribute with the
     * same name exists but is not a vector of number, null is returned.
     * 
     * @param key
     *            The name of the number to search.
     * @return The vector of numbers or null if not found.
     */
    public native ArrayList<? extends Number> getVector(String key);
    
    /**
     * Get the array of objects bound to key. Arrays of objects are special
     * attributes whose value is a sequence of objects. If an attribute with the
     * same name exists but is not an array, null is returned.
     * 
     * @param key
     *            The name of the array to search.
     * @return The array of objects or null if not found.
     */
    public native Object[] getArray(String key);
    
    /**
     * Get the hash bound to key. Hashes are special attributes whose value is a
     * set of pairs (name,object). Instances of object implementing the
     * {@link CompoundAttribute} interface are considered like hashes since they
     * can be transformed to a hash. If an attribute with the same name exists
     * but is not a hash, null is returned. We cannot enforce the type of the
     * key. It is considered a string and you should use "Object.toString()" to
     * get it.
     * 
     * @param key
     *            The name of the hash to search.
     * @return The hash or null if not found.
     */
    public native HashMap<?, ?> getHash(String key);
    
    /**
     * Does this element store a value for the given attribute key?
     * 
     * @param key
     *            The name of the attribute to search.
     * @return True if a value is present for this attribute.
     */
    public native boolean hasAttribute(String key);
    
    /**
     * Does this element store a value for the given attribute key and this
     * value is an instance of the given class?
     * 
     * @param key
     *            The name of the attribute to search.
     * @param clazz
     *            The expected class of the attribute value.
     * @return True if a value is present for this attribute.
     */
    public native boolean hasAttribute(String key, Class<?> clazz);
    
    /**
     * Does this element store a label value for the given key? A label is an
     * attribute whose value is a string.
     * 
     * @param key
     *            The name of the label.
     * @return True if a value is present for this attribute and implements
     *         CharSequence.
     */
    public native boolean hasLabel(String key);
    
    /**
     * Does this element store a number for the given key? A number is an
     * attribute whose value is an instance of Number.
     * 
     * @param key
     *            The name of the number.
     * @return True if a value is present for this attribute and can contain a
     *         double (inherits from Number).
     */
    public native boolean hasNumber(String key);
    
    /**
     * Does this element store a vector value for the given key? A vector is an
     * attribute whose value is a sequence of numbers.
     * 
     * @param key
     *            The name of the vector.
     * @return True if a value is present for this attribute and can contain a
     *         sequence of numbers.
     */
    public native boolean hasVector(String key);
    
    /**
     * Does this element store an array value for the given key? A vector is an
     * attribute whose value is an array of objects.
     * 
     * @param key
     *            The name of the array.
     * @return True if a value is present for this attribute and can contain an
     *         array object.
     */
    public native boolean hasArray(String key);
    
    /**
     * Does this element store a hash value for the given key? A hash is a set
     * of pairs (key,value) or objects that implement the
     * {@link org.graphstream.graph.CompoundAttribute} class.
     * 
     * @param key
     *            The name of the hash.
     * @return True if a value is present for this attribute and can contain a
     *         hash.
     */
    public native boolean hasHash(String key);
    
    /**
     * Iterator on all attributes keys.
     * 
     * @return An iterator on the key set of attributes.
     */
    public native Iterator<String> getAttributeKeyIterator();
    
    /**
     * An iterable view on the set of attributes keys usable with the for-each
     * loop.
     * 
     * @return an iterable view on each attribute key, null if there are no
     *         attributes.
     */
    public native Iterable<String> getAttributeKeySet();
    
    /**
     * Remove all registered attributes. This includes numbers, labels and
     * vectors.
     */
    public native void clearAttributes();
    
    /**
     * Add or replace the value of an attribute. Existing attributes are
     * overwritten silently. All classes inheriting from Number can be
     * considered as numbers. All classes inheriting from CharSequence can be
     * considered as labels. You can pass zero, one or more arguments for the
     * attribute values. If no value is given, a boolean with value "true" is
     * added. If there is more than one value, an array is stored. If there is
     * only one value, the value is stored (but not in an array).
     * 
     * @param attribute
     *            The attribute name.
     * @param values
     *            The attribute value or set of values.
     */
    public native void addAttribute(String attribute, Object... values);
    
    /**
     * Like {@link #addAttribute(String, Object...)} but for consistency.
     * 
     * @param attribute
     *            The attribute name.
     * @param values
     *            The attribute value or array of values.
     * @see #addAttribute(String, Object...)
     */
    public native void changeAttribute(String attribute, Object... values);
    
    /**
     * Like {@link #addAttribute(String, Object...)} but for consistency.
     * 
     * @param attribute
     *            The attribute name.
     * @param values
     *            The attribute value or array of values.
     * @see #addAttribute(String, Object...)
     */
    public native void setAttribute(String attribute, Object... values);
    
    /**
     * Add or replace each attribute found in attributes. Existing attributes
     * are overwritten silently. All classes inheriting from Number can be
     * considered as numbers. All classes inheriting from CharSequence can be
     * considered as labels.
     * 
     * @param attributes
     *            A set of (key,value) pairs.
     */
    public native void addAttributes(Map<String, Object> attributes);
    
    /**
     * Remove an attribute. Non-existent attributes errors are ignored silently.
     * 
     * @param attribute
     *            Name of the attribute to remove.
     */
    public native void removeAttribute(String attribute);
    
    /**
     * Number of attributes stored in this element.
     * 
     * @return the number of attributes.
     */
    public native int getAttributeCount();
}
