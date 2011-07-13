
#ifndef _GRAPHSTREAM_GRAPH_H_
#define _GRAPHSTREAM_GRAPH_H_

#include "gs_types.h"
#include "gs_common.h"
#include "gs_element.h"
#include "gs_node.h"
#include "gs_edge.h"
#include "gs_stream.h"

struct _graph {
  element_t parent;
  Eina_Hash *nodes;
  Eina_Hash *edges;
  GS_SOURCE_FIELD;
  GS_SINK_FIELD;
};

#define GS_GRAPH(e) ((graph_t*)CHECK_TYPE(e,GRAPH_TYPE))

GSAPI graph_t* gs_graph_create(const element_id_t);

GSAPI void gs_graph_destroy(graph_t*);

GSAPI node_t *gs_graph_node_add(const graph_t*,
				const element_id_t);

GSAPI node_t *gs_graph_node_get(const graph_t*,
				const element_id_t);

GSAPI void gs_graph_node_delete(const graph_t*,
				const element_id_t);

GSAPI edge_t *gs_graph_edge_add(const graph_t*,
				const element_id_t,
				const element_id_t,
				const element_id_t,
				bool_t);

GSAPI edge_t *gs_graph_edge_get(const graph_t*,
				const element_id_t);

GSAPI void gs_graph_edge_delete(const graph_t*,
				const element_id_t);

GSAPI iterator_t *gs_graph_node_iterator_new(const graph_t*);

GSAPI void gs_graph_node_foreach(const graph_t *graph,
				 const node_cb_t callback,
				 void **data);

GSAPI iterator_t *gs_graph_edge_iterator_new(const graph_t*);

GSAPI void gs_graph_edge_foreach(const graph_t *graph,
				 const edge_cb_t callback,
				 void **data);


#endif /* _GRAPHSTREAM_GRAPH_H_ */
