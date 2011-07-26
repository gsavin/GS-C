#ifndef _GS_NODE_H_
#define _GS_NODE_H_

#include "gs_types.h"
#include "gs_common.h"
#include "gs_element.h"
#include "gs_edge.h"

#ifdef __cplusplus
extern "C" {
#endif

struct _node {
  GSElement parent;
  GSGraph  *graph;
  uint      edge_count;
  GList    *edges;
};

#define GS_NODE(e) ((GSNode*)CHECK_TYPE(e,NODE_TYPE))

GSAPI GSNode* gs_node_create(GSGraph   *graph,
			     const gsid id);

GSAPI void gs_node_destroy(GSNode *node);

GSAPI void gs_node_edge_register(GSNode *node,
				 GSEdge *edge);

GSAPI GSIterator* gs_node_edge_iterator_new(const GSNode *node);

#ifdef __cplusplus
}
#endif

#endif /* _GS_NODE_H_ */
