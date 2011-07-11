#ifndef _GS_EDGE_H_
#define _GS_EDGE_H_

#include "gs_types.h"
#include "gs_common.h"
#include "gs_element.h"

struct _edge {
  element_t parent;
  graph_t *graph;
  node_t *source;
  node_t *target;
  bool_t directed;
};

#define GS_EDGE(e) ((edge_t*)CHECK_TYPE(e,EDGE_TYPE))

GSAPI edge_t* gs_edge_create(const graph_t*,
			     const element_id_t,
			     const node_t*,
			     const node_t*,
			     bool_t);

GSAPI void gs_edge_destroy(edge_t*);


#endif /* _GS_EDGE_H_ */
