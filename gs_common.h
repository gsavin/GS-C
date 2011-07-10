#ifndef _GS_COMMON_H_
#define _GS_COMMON_H_

#include "config.h"
#include "gs_types.h"
#include "error.h"

typedef struct {
  int type;
} object_t;

#include "magic.h"

GSAPI int gs_init();
GSAPI void gs_shutdown();

GSAPI void gs_iterator_free(iterator_t*);

#endif /* _GS_COMMON_H_ */
