#include <string.h>
#include "gs_native_common.h"
#include "org_graphstream_util_NativePointer.h"

GSAPI inline gpointer
gs_native_get_element(JNIEnv *env,
		      jobject obj)
{
  jfieldID fid;
  jclass   cls;
  jobject  ptr;
  gpointer ref;

  cls = (*env)->GetObjectClass(env, obj);
  fid = (*env)->GetFieldID(env, cls, "__ref", "Lorg/graphstream/util/NativePointer;");
  ptr = (*env)->GetObjectField(env, obj, fid);
  ref = Java_org_graphstream_util_NativePointer_getPointer(env, ptr);

  fprintf(stdout, "get ref 0x%lx\n", ref);

  return ref;
}

GSAPI void
gs_native_set_element(JNIEnv    *env,
		      jobject    obj,
		      gpointer   element)
{
  jfieldID fid;
  jclass   cls;
  jobject  ref;

  cls = (*env)->GetObjectClass(env, obj);
  fid = (*env)->GetFieldID(env, cls, "__ref", "Lorg/graphstream/util/NativePointer;");
  ref = (*env)->GetObjectField(env, obj, fid);
  
  Java_org_graphstream_util_NativePointer_setPointer(env, ref, element);
}

GSAPI void
gs_native_copy_string(JNIEnv *env,
		      jstring str,
		      char  **dest)
{
  long  l;
  long  i;
  
  l       = (*env)->GetStringUTFLength(env, str);
  (*dest) = (char*) malloc((1+l)*sizeof(char));

  (*env)->GetStringUTFRegion(env, str, 0, l, (*dest));
  
  *((*dest)+l) = '\0';
}
