#ifndef _GS_NATIVE_COMMON_H_
#define _GS_NATIVE_COMMON_H_

#include <jni.h>
#include "gs_element.h"

#ifdef __cplusplus
extern "C" {
#endif

  /**
   * Get the native ressource GSElement associate with a java
   * object. This object should be a NativeElement.
   *
   * @param env the JNI environment
   * @param obj this object
   *
   * @return a GSElement
   */
  GSAPI gpointer
  gs_native_get_element(JNIEnv *env,
			jobject obj);

  GSAPI void
  gs_native_set_element(JNIEnv    *env,
			jobject    obj,
			gpointer   element);

  GSAPI void
  gs_native_copy_string(JNIEnv *env,
			jstring str,
			char  **dest);
  
#ifdef __cplusplus
}
#endif

#endif /* _GS_NATIVE_COMMON_H_ */
