#include "gs_graph.h"
#include "gs_id.h"

#ifndef GRAPH_ELEMENT_HASH_FUNCTION
#define GRAPH_ELEMENT_HASH_FUNCTION eina_hash_string_djb2_new
#endif

/**********************************************************************
 * PRIVATE
 */

GSAPI static void
_gs_graph_node_destroy(void *data)
{
  element_id_t id;

  id = gs_element_id_get(GS_ELEMENT(data));
  gs_node_destroy(GS_NODE(data));
  gs_id_release(id);
}

GSAPI static void
_gs_graph_edge_destroy(void *data)
{
  element_id_t id;

  id = gs_element_id_get(GS_ELEMENT(data));
  gs_edge_destroy(GS_EDGE(data));
  gs_id_release(id);
}

GSAPI static void
_gs_graph_sink_callback(const sink_t *sink,
			event_t event,
			size_t size,
			const void **data)
{
  graph_t *g;
  g = (graph_t*) sink->container;

  switch(event) {
  case NODE_ADDED:
    gs_graph_node_add(g, (element_id_t) data[1]);
    break;
  case NODE_DELETED:
    gs_graph_node_delete(g, (element_id_t) data[1]);
    break;
  case EDGE_ADDED:
    gs_graph_edge_add(g,
		      (element_id_t) data[1],
		      (element_id_t) data[2],
		      (element_id_t) data[3],
		      (bool_t) data[4]);
    break;
  case EDGE_DELETED:
    gs_graph_edge_delete(g,
			 (element_id_t) data[1]);
    break;
  default:
    break;
  }
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

  gs_element_init(GS_ELEMENT(graph),
		  id);

  gs_stream_source_init(GS_SOURCE(graph),
			id);

  gs_stream_sink_init(GS_SINK(graph),
		      graph,
		      GS_SINK_CALLBACK(_gs_graph_sink_callback));

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
    element_id_t nid;
    nid = gs_id_copy(id);
    node = gs_node_create(graph, nid);
    eina_hash_add(graph->nodes, nid, node);
  }

  gs_stream_source_trigger_node_added(GS_SOURCE(graph),
				      gs_element_id_get(GS_ELEMENT(graph)),
				      id);

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
  gs_stream_source_trigger_node_deleted(GS_SOURCE(graph),
				      gs_element_id_get(GS_ELEMENT(graph)),
				      id);

  eina_hash_del_by_key(graph->nodes, id);
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
    element_id_t eid;

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

    eid = gs_id_copy(id);
    edge = gs_edge_create(graph, eid, src, trg, directed);
    eina_hash_add(graph->edges, eid, edge);

    gs_node_edge_register(src, edge);
    gs_node_edge_register(trg, edge);
  }

  gs_stream_source_trigger_edge_added(GS_SOURCE(graph),
				      gs_element_id_get(GS_ELEMENT(graph)),
				      id,
				      id_src,
				      id_trg,
				      directed);

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

  gs_stream_source_trigger_edge_deleted(GS_SOURCE(graph),
					gs_element_id_get(GS_ELEMENT(graph)),
					id);

  eina_hash_del_by_key(graph->edges, id);
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
