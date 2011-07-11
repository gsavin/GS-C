#include "gs_graph.h"

#ifndef GRAPH_ELEMENT_HASH_FUNCTION
#define GRAPH_ELEMENT_HASH_FUNCTION eina_hash_string_djb2_new
#endif

/**********************************************************************
 * PRIVATE
 */

GSAPI static void
_gs_graph_node_destroy(void *data)
{
  gs_node_destroy(GS_NODE(data));
}

GSAPI static void
_gs_graph_edge_destroy(void *data)
{
  gs_edge_destroy(GS_EDGE(data));
}

GSAPI static void
_gs_graph_sink_callback(const sink_t *sink,
			event_t event,
			const void **data)
{

}

/**********************************************************************
 * PUBLIC
 */

GSAPI graph_t*
gs_graph_create(const element_id_t id)
{
  graph_t *graph;
  graph = (graph_t*) malloc(sizeof(graph_t));
  
  GS_OBJECT(graph)->type = GRAPH_TYPE;
  gs_element_init(GS_ELEMENT(graph),id);
  gs_stream_source_init(GS_SOURCE(graph), id);
  gs_stream_sink_init(GS_SINK(graph), GS_SINK_CALLBACK(_gs_graph_sink_callback));

  graph->nodes = GRAPH_ELEMENT_HASH_FUNCTION(_gs_graph_node_destroy);
  graph->edges = GRAPH_ELEMENT_HASH_FUNCTION(_gs_graph_edge_destroy);

  return graph;
}

GSAPI void
gs_graph_destroy(graph_t *graph)
{
  gs_element_finalize(GS_ELEMENT(graph));
  gs_stream_source_finalize(GS_SOURCE(graph));
  gs_stream_sink_finalize(GS_SINK(graph));

  eina_hash_free(graph->edges);
  eina_hash_free(graph->nodes);

  free(graph);
}

GSAPI node_t*
gs_graph_node_add(const graph_t *graph,
		  const element_id_t id)
{
  node_t *node;
  node = eina_hash_find(graph->nodes, id);

  if(node != NULL) {
#ifdef GRAPH_STRICT_CHECKING
    ERROR(GS_ERROR_ID_ALREADY_IN_USE);
#endif
  }
  else {
    node = gs_node_create(graph, id);
    eina_hash_add(graph->nodes, id, node);
  }

  return node;
}

GSAPI node_t*
gs_graph_node_get(const graph_t *graph,
		  const element_id_t id)
{
  node_t *node;
  node = eina_hash_find(graph->nodes, id);

  return node;
}

GSAPI void
gs_graph_node_delete(const graph_t *graph,
		     const element_id_t id)
{
  eina_hash_del(graph->nodes, id, NULL);
}

GSAPI edge_t*
gs_graph_edge_add(const graph_t *graph,
		  const element_id_t id,
		  const element_id_t id_src,
		  const element_id_t id_trg,
		  bool_t directed)
{
  edge_t *edge;
  edge = eina_hash_find(graph->edges, id);

  if(edge != NULL) {
#ifdef GRAPH_STRICT_CHECKING
    ERROR(GS_ERROR_ID_ALREADY_IN_USE);
#endif
  }
  else {
    node_t *src, *trg;

    src = gs_graph_node_get(graph, id_src);
    trg = gs_graph_node_get(graph, id_trg);

    if(src == NULL || trg == NULL) {
#ifdef GRAPH_AUTOCREATE
      if(src == NULL)
	src = gs_graph_node_add(graph, id_src);
      
      if(trg == NULL)
	trg = gs_graph_node_add(graph, id_trg);
#else
      ERROR(GS_ERROR_NODE_NOT_FOUND);
#endif
    }

    edge = gs_edge_create(graph, id, src, trg, directed);
    eina_hash_add(graph->edges, id, edge);
  }

  return edge;
}

GSAPI edge_t*
gs_graph_edge_get(const graph_t *graph,
		  const element_id_t id)
{
  edge_t *edge;
  edge = eina_hash_find(graph->edges, id);
  
  return edge;
}

GSAPI void
gs_graph_edge_delete(const graph_t *graph,
		     const element_id_t id)
{
  eina_hash_del(graph->edges, id, NULL);
}

GSAPI iterator_t *
gs_graph_node_iterator_new(const graph_t *graph)
{
  iterator_t *it;
  it = (iterator_t*) eina_hash_iterator_data_new(graph->nodes);

  if(it == NULL) {
    ERROR(GS_ERROR_UNKNOWN);
  }

  return it;
}

GSAPI node_t *
gs_graph_node_iterator_next(iterator_t *it)
{
  node_t *next;

  if(eina_iterator_next(it, (void**) &next) != EINA_TRUE)
    return NULL;

  return next;
}

GSAPI void
gs_graph_node_foreach(const graph_t *graph,
		      const node_cb_t callback,
		      void **data)
{
  iterator_t *it;
  node_t *node;

  it = gs_graph_node_iterator_new(graph);
  EINA_ITERATOR_FOREACH(it, node) callback(node, data);
  gs_iterator_free(it);
}

GSAPI iterator_t *
gs_graph_edge_iterator_new(const graph_t *graph)
{
  iterator_t *it;
  it = (iterator_t*) eina_hash_iterator_data_new(graph->edges);

  if(it == NULL) {
    ERROR(GS_ERROR_UNKNOWN);
  }

  return it;
}

GSAPI edge_t *
gs_graph_edge_iterator_next(iterator_t *it)
{
  edge_t *next;

  if(eina_iterator_next(it, (void**) &next) != EINA_TRUE)
    return NULL;

  return next;
}

GSAPI void
gs_graph_edge_foreach(const graph_t *graph,
		      const edge_cb_t callback,
		      void **data)
{
  iterator_t *it;
  edge_t *edge;

  it = gs_graph_edge_iterator_new(graph);
  EINA_ITERATOR_FOREACH(it, edge) callback(edge, data);
  gs_iterator_free(it);
}
