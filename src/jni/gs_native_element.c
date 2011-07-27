#include <jni.h>

#include "gs_element.h"
#include "org_graphstream_graph_implementations_NativeElement.h"

#define NATIVE_ELEMENT(m) Java_org_graphstream_graph_implementations_NativeElement_ ## m

/*
 * Class:     org_graphstream_graph_implementations_NativeElement
 * Method:    init
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL
NATIVE_ELEMENT(init)(JNIEnv *env,
		     jobject obj)
{
  GSElement  *element;
  char       *nativeId;
  jstring     id;
  jclass      cls;
  jfieldID    fid;

  cls = (*env)->GetObjectClass(env, obj);
  fid = (*env)->GetFieldID(env, cls, "id", "Ljava/lang/String;");
  id  = (*env)->GetObjectField(env, obj, fid);

  if (id == NULL) {
    fprintf(stderr, "Error : id is null\n");
    exit(1);
  }

  gs_native_copy_string(env, id, &nativeId);
  
  if (nativeId == NULL) {
    fprintf(stderr, "Error : failed to copy id\n");
    exit(1);
  }
  
  element = (GSElement*) malloc(sizeof(GSElement));
  gs_element_init(element, nativeId);

  fprintf(stdout, "New native element %s (0x%lx)\n", nativeId, element);

  /*
   * Set the pointer to the native element.
   */
  gs_native_set_element(env, obj, element, sizeof(GSElement));
}

/*
 * Class:     org_graphstream_graph_implementations_NativeElement
 * Method:    uninit
 * Signature: ()V
 */
JNIEXPORT void JNICALL
NATIVE_ELEMENT(uninit)(JNIEnv *env,
		       jobject obj)
{
  GSElement *element;
  char      *id;
  gpointer   ref;
  
  ref = gs_native_get_element(env, obj);

  fprintf(stdout, "try free element 0x%lx\n", ref);
  element = ref;

  if (element == NULL) {
    fprintf(stderr, "no element pointer\n");
    return;
  }

  //id = gs_element_id_get(element);
  
  //printf("free element \"%s\"\n", id);

  //gs_element_finalize(element);
  //gs_id_release(id);

  //free(element);
}

/*
 * Class:     org_graphstream_graph_implementations_NativeElement
 * Method:    getAttribute
 * Signature: (Ljava/lang/String;)Ljava/lang/Object;
 */
JNIEXPORT jobject JNICALL
NATIVE_ELEMENT(getAttribute__Ljava_lang_String_2)(JNIEnv *env,
						  jobject obj,
						  jstring str)
{
  return NULL;
}

/*
 * Class:     org_graphstream_graph_implementations_NativeElement
 * Method:    getFirstAttributeOf
 * Signature: ([Ljava/lang/String;)Ljava/lang/Object;
 */
JNIEXPORT jobject JNICALL
NATIVE_ELEMENT(getFirstAttributeOf___3Ljava_lang_String_2)(JNIEnv      *env,
							   jobject      obj,
							   jobjectArray array)
{
  return NULL;
}

/*
 * Class:     org_graphstream_graph_implementations_NativeElement
 * Method:    getAttribute
 * Signature: (Ljava/lang/String;Ljava/lang/Class;)Ljava/lang/Object;
 */
JNIEXPORT jobject JNICALL
NATIVE_ELEMENT(getAttribute__Ljava_lang_String_2Ljava_lang_Class_2)(JNIEnv *env,
								    jobject obj,
								    jstring str,
								    jclass  cls)
{
  return NULL;
}

/*
 * Class:     org_graphstream_graph_implementations_NativeElement
 * Method:    getFirstAttributeOf
 * Signature: (Ljava/lang/Class;[Ljava/lang/String;)Ljava/lang/Object;
 */
JNIEXPORT jobject JNICALL
NATIVE_ELEMENT(getFirstAttributeOf__Ljava_lang_Class_2_3Ljava_lang_String_2)(JNIEnv      *env,
									     jobject      obj,
									     jclass       cls,
									     jobjectArray array)
{
  return NULL;
}

/*
 * Class:     org_graphstream_graph_implementations_NativeElement
 * Method:    getLabel
 * Signature: (Ljava/lang/String;)Ljava/lang/CharSequence;
 */
JNIEXPORT jobject JNICALL
NATIVE_ELEMENT(getLabel)(JNIEnv *env,
			 jobject obj,
			 jstring key)
{

}

/*
 * Class:     org_graphstream_graph_implementations_NativeElement
 * Method:    getNumber
 * Signature: (Ljava/lang/String;)D
 */
JNIEXPORT jdouble JNICALL
NATIVE_ELEMENT(getNumber)(JNIEnv *env,
			  jobject obj,
			  jstring key)
{
  return 0.0;
}

/*
 * Class:     org_graphstream_graph_implementations_NativeElement
 * Method:    getVector
 * Signature: (Ljava/lang/String;)Ljava/util/ArrayList;
 */
JNIEXPORT jobject JNICALL
NATIVE_ELEMENT(getVector)(JNIEnv *env,
			  jobject obj,
			  jstring key)
{
  return NULL;
}

/*
 * Class:     org_graphstream_graph_implementations_NativeElement
 * Method:    getArray
 * Signature: (Ljava/lang/String;)[Ljava/lang/Object;
 */
JNIEXPORT jobjectArray JNICALL
NATIVE_ELEMENT(getArray)(JNIEnv *env,
			 jobject obj,
			 jstring key)
{
  return NULL;
}

/*
 * Class:     org_graphstream_graph_implementations_NativeElement
 * Method:    getHash
 * Signature: (Ljava/lang/String;)Ljava/util/HashMap;
 */
JNIEXPORT jobject JNICALL
NATIVE_ELEMENT(getHash)(JNIEnv *env,
			jobject obj,
			jstring key)
{

}

/*
 * Class:     org_graphstream_graph_implementations_NativeElement
 * Method:    hasAttribute
 * Signature: (Ljava/lang/String;)Z
 */
JNIEXPORT jboolean JNICALL
NATIVE_ELEMENT(hasAttribute__Ljava_lang_String_2)(JNIEnv *env,
						  jobject obj,
						  jstring key)
{
  return GS_FALSE;
}

/*
 * Class:     org_graphstream_graph_implementations_NativeElement
 * Method:    hasAttribute
 * Signature: (Ljava/lang/String;Ljava/lang/Class;)Z
 */
JNIEXPORT jboolean JNICALL
NATIVE_ELEMENT(hasAttribute__Ljava_lang_String_2Ljava_lang_Class_2)(JNIEnv *env,
								    jobject obj,
								    jstring key,
								    jclass  cls)
{
  return GS_FALSE;
}

/*
 * Class:     org_graphstream_graph_implementations_NativeElement
 * Method:    hasLabel
 * Signature: (Ljava/lang/String;)Z
 */
JNIEXPORT jboolean JNICALL NATIVE_ELEMENT(hasLabel)(JNIEnv *env,
						    jobject obj,
						    jstring key)
{
  return GS_FALSE;
}

/*
 * Class:     org_graphstream_graph_implementations_NativeElement
 * Method:    hasNumber
 * Signature: (Ljava/lang/String;)Z
 */
JNIEXPORT jboolean JNICALL NATIVE_ELEMENT(hasNumber)(JNIEnv *env,
						     jobject obj,
						     jstring key)
{

}

/*
 * Class:     org_graphstream_graph_implementations_NativeElement
 * Method:    hasVector
 * Signature: (Ljava/lang/String;)Z
 */
JNIEXPORT jboolean JNICALL NATIVE_ELEMENT(hasVector)(JNIEnv *env,
						     jobject obj,
						     jstring key)
{

}

/*
 * Class:     org_graphstream_graph_implementations_NativeElement
 * Method:    hasArray
 * Signature: (Ljava/lang/String;)Z
 */
JNIEXPORT jboolean JNICALL NATIVE_ELEMENT(hasArray)(JNIEnv *env,
						    jobject obj,
						    jstring key)
{

}

/*
 * Class:     org_graphstream_graph_implementations_NativeElement
 * Method:    hasHash
 * Signature: (Ljava/lang/String;)Z
 */
JNIEXPORT jboolean JNICALL NATIVE_ELEMENT(hasHash)(JNIEnv *env,
						   jobject obj,
						   jstring key)
{

}

/*
 * Class:     org_graphstream_graph_implementations_NativeElement
 * Method:    getAttributeKeyIterator
 * Signature: ()Ljava/util/Iterator;
 */
JNIEXPORT jobject JNICALL NATIVE_ELEMENT(getAttributeKeyIterator)(JNIEnv *env,
								  jobject obj)
{

}

/*
 * Class:     org_graphstream_graph_implementations_NativeElement
 * Method:    getAttributeKeySet
 * Signature: ()Ljava/lang/Iterable;
 */
JNIEXPORT jobject JNICALL NATIVE_ELEMENT(getAttributeKeySet)(JNIEnv *env,
							     jobject obj)
{
  return NULL;
}

/*
 * Class:     org_graphstream_graph_implementations_NativeElement
 * Method:    clearAttributes
 * Signature: ()V
 */
JNIEXPORT void JNICALL NATIVE_ELEMENT(clearAttributes)(JNIEnv *env,
						       jobject obj)
{

}

/*
 * Class:     org_graphstream_graph_implementations_NativeElement
 * Method:    addAttribute
 * Signature: (Ljava/lang/String;[Ljava/lang/Object;)V
 */
JNIEXPORT void JNICALL NATIVE_ELEMENT(addAttribute)(JNIEnv      *env,
						    jobject      obj,
						    jstring      key,
						    jobjectArray array)
{

}

/*
 * Class:     org_graphstream_graph_implementations_NativeElement
 * Method:    changeAttribute
 * Signature: (Ljava/lang/String;[Ljava/lang/Object;)V
 */
JNIEXPORT void JNICALL NATIVE_ELEMENT(changeAttribute)(JNIEnv      *env,
						       jobject      obj,
						       jstring      key,
						       jobjectArray array)
{

}

/*
 * Class:     org_graphstream_graph_implementations_NativeElement
 * Method:    setAttribute
 * Signature: (Ljava/lang/String;[Ljava/lang/Object;)V
 */
JNIEXPORT void JNICALL NATIVE_ELEMENT(setAttribute)(JNIEnv      *env,
						    jobject      obj,
						    jstring      key,
						    jobjectArray array)
{

}

/*
 * Class:     org_graphstream_graph_implementations_NativeElement
 * Method:    addAttributes
 * Signature: (Ljava/util/Map;)V
 */
JNIEXPORT void JNICALL NATIVE_ELEMENT(addAttributes)(JNIEnv *env,
						     jobject obj,
						     jobject attr)
{

}

/*
 * Class:     org_graphstream_graph_implementations_NativeElement
 * Method:    removeAttribute
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL NATIVE_ELEMENT(removeAttribute)(JNIEnv *env,
						       jobject obj,
						       jstring key)
{

}

/*
 * Class:     org_graphstream_graph_implementations_NativeElement
 * Method:    getAttributeCount
 * Signature: ()I
 */
JNIEXPORT jint JNICALL NATIVE_ELEMENT(getAttributeCount)(JNIEnv *env,
							 jobject obj)
{
  return 0;
}
