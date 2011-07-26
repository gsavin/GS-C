#ifndef _GS_EDGE_H_
#define _GS_EDGE_H_

#include "gs_types.h"
#include "gs_common.h"
#include "gs_element.h"

#ifdef __cplusplus
extern "C" {
#endif

struct _edge {
  GSElement parent;
  GSGraph  *graph;
  GSNode   *source;
  GSNode   *target;
  gsboolean directed;
};

#define GS_EDGE(e) ((GSEdge*)CHECK_TYPE(e,EDGE_TYPE))

GSAPI GSEdge* gs_edge_create(GSGraph    *graph,
			     const gsid  id,
			     GSNode     *source,
			     GSNode     *target,
			     gsboolean   directed);

GSAPI void gs_edge_destroy(GSEdge *edge);

#define GS_EDGE_OPOSITE_OF(e,n)			\
  (e->source == n ? e->target : e->source)

#ifdef __cplusplus
}
#endif

#endif /* _GS_EDGE_H_ */
