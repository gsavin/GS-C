#include <jni.h>
#include "org_graphstream_util_NativePointer.h"

#define NATIVE_POINTER(m) Java_org_graphstream_util_NativePointer_ ## m

/*
 * Class:     org_graphstream_util_NativePointer
 * Method:    getPointer
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL
NATIVE_POINTER(getPointer)(JNIEnv *env,
			   jobject obj)
{ 
  long     p32;
  long     p64;
  jfieldID fid;
  jclass   cls;
  jlong    ref;

  cls = (*env)->GetObjectClass(env, obj);
  fid = (*env)->GetFieldID(env, cls, "__32bits", "J");
  p32 = (*env)->GetLongField(env, obj, fid);
  fid = (*env)->GetFieldID(env, cls, "__64bits", "J");
  p64 = (*env)->GetLongField(env, obj, fid);
  ref =  (p64 << 32) | (p32 & 0x00000000FFFFFFFF);

  fprintf(stdout, "divided :\n p32 : 0x%lx\n p64 : 0x%lx\n ref : 0x%lx\n", p32, p64,ref);

  return ref;
}

/*
 * Class:     org_graphstream_util_NativePointer
 * Method:    setPointer
 * Signature: (J)V
 */
JNIEXPORT void JNICALL
NATIVE_POINTER(setPointer)(JNIEnv *env,
			   jobject obj,
			   jlong   ref)
{
  long     p32;
  long     p64;
  jfieldID fid;
  jclass   cls;

  p32 = ref & 0x00000000FFFFFFFF;
  p64 = ( ref >> 32 ) & 0x00000000FFFFFFFF;

  fprintf(stdout, "merge :\n p32 : 0x%lx\n p64 : 0x%lx\n ref : 0x%lx\n", p32, p64,ref);

  cls = (*env)->GetObjectClass(env, obj);
  fid = (*env)->GetFieldID(env, cls, "__32bits", "J");
  
  (*env)->SetLongField(env, obj, fid, p32);

  fid = (*env)->GetFieldID(env, cls, "__64bits", "J");
  
  (*env)->SetLongField(env, obj, fid, p64);
}
