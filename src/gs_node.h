#ifndef _GS_NODE_H_
#define _GS_NODE_H_

#include "gs_types.h"
#include "gs_common.h"
#include "gs_element.h"
#include "gs_edge.h"

struct _node {
  element_t parent;
  graph_t *graph;
  uint edge_count;
  Eina_List *edges;
};

#define GS_NODE(e) ((node_t*)CHECK_TYPE(e,NODE_TYPE))

GSAPI node_t* gs_node_create(const graph_t*,
			     const element_id_t);

GSAPI void gs_node_destroy(node_t*);

GSAPI void gs_node_edge_register(node_t *node,
				 const edge_t * edge);

GSAPI iterator_t* gs_node_edge_iterator_new(const node_t *node);

#endif /* _GS_NODE_H_ */
