#ifndef _MAGIC_H_
#define _MAGIC_H_

#include "config.h"
#include "gs_common.h"

#define ELEMENT_TYPE 0xE8760000
#define NODE_TYPE 0xE8760001
#define EDGE_TYPE 0xE8760002
#define GRAPH_TYPE 0xE8760004

#define GS_OBJECT(p) ((GSObject*)p)
#define CHECK_TYPE(e,t)	gs_safe_cast(e,t)

#ifdef __cplusplus
extern "C" {
#endif

GSAPI static inline void *
gs_safe_cast(void *p, int type)
{
  if ((GS_OBJECT(p)->type & type) != type)
    ERROR(GS_ERROR_INVALID_TYPE);

  return  p;
}

#ifdef __cplusplus
}
#endif

#endif /* _MAGIC_H_ */
