#ifndef _GS_COMMON_H_
#define _GS_COMMON_H_

#include "config.h"
#include "gs_types.h"
#include "error.h"
#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int type;
} GSObject;

#include "magic.h"

GSAPI int gs_init();
GSAPI void gs_shutdown();

GSAPI void gs_iterator_free(GSIterator *it);

GSAPI GSNode *gs_iterator_next_node(GSIterator *it);
GSAPI GSEdge *gs_iterator_next_edge(GSIterator *it);


#ifdef __cplusplus
}
#endif

#endif /* _GS_COMMON_H_ */
