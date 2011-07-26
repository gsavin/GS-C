
#ifndef _GRAPHSTREAM_GRAPH_H_
#define _GRAPHSTREAM_GRAPH_H_

#include "gs_types.h"
#include "gs_common.h"
#include "gs_element.h"
#include "gs_node.h"
#include "gs_edge.h"
#include "gs_stream.h"

#ifdef __cplusplus
extern "C" {
#endif

struct _graph {
  GSElement   parent;
  GHashTable *nodes;
  GHashTable *edges;

  GS_SOURCE_FIELD;
  GS_SINK_FIELD;
};

#define GS_GRAPH(e) ((GSGraph*)CHECK_TYPE(e,GSGRAPHYPE))

GSAPI GSGraph* gs_graph_create(const gsid);

GSAPI void gs_graph_destroy(GSGraph *graph);

GSAPI GSNode *gs_graph_node_add(GSGraph   *graph,
				const gsid id);

GSAPI GSNode *gs_graph_node_get(const GSGraph *graph,
				const gsid     id);

GSAPI void gs_graph_node_delete(const GSGraph *graph,
				const gsid     id);

GSAPI GSEdge *gs_graph_edge_add(GSGraph   *graph,
				const gsid id,
				const gsid source,
				const gsid target,
				gsboolean  directed);

GSAPI GSEdge *gs_graph_edge_get(const GSGraph *graph,
				const gsid     id);

GSAPI void gs_graph_edge_delete(const GSGraph *graph,
				const gsid     id);

GSAPI GSIterator *gs_graph_node_iterator_new(const GSGraph *graph);

GSAPI void gs_graph_node_foreach(const GSGraph  *graph,
				 const GSNodeCB  callback,
				 void           *data);

GSAPI GSIterator *gs_graph_edge_iterator_new(const GSGraph *graph);

GSAPI void gs_graph_edge_foreach(const GSGraph *graph,
				 const GSEdgeCB callback,
				 void          *data);

#ifdef __cplusplus
}
#endif

#endif /* _GRAPHSTREAM_GRAPH_H_ */
