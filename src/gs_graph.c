#include "gs_graph.h"
#include "gs_id.h"

/**********************************************************************
 * PRIVATE
 */

GSAPI static void
_gs_graph_node_destroy(void *data)
{
  gsid id;

  id = gs_element_id_get(GS_ELEMENT(data));
  gs_node_destroy(GS_NODE(data));
  gs_id_release(id);
}

GSAPI static void
_gs_graph_edge_destroy(void *data)
{
  gsid id;

  id = gs_element_id_get(GS_ELEMENT(data));
  gs_edge_destroy(GS_EDGE(data));
  gs_id_release(id);
}

GSAPI static void
_gs_graph_sink_callback(const GSSink *sink,
			event_t       event,
			size_t        size,
			const void  **data)
{
  GSGraph *g;
  g = (GSGraph*) sink->container;

  switch(event) {
  case NODE_ADDED:
    gs_graph_node_add(g, (gsid) data[1]);
    break;
  case NODE_DELETED:
    gs_graph_node_delete(g, (gsid) data[1]);
    break;
  case EDGE_ADDED:
    gs_graph_edge_add(g,
		      (gsid)      data[1],
		      (gsid)      data[2],
		      (gsid)      data[3],
		      (gsboolean) data[4]);
    break;
  case EDGE_DELETED:
    gs_graph_edge_delete(g,
			 (gsid) data[1]);
    break;
  default:
    break;
  }
}

/**********************************************************************
 * PUBLIC
 */

GSAPI GSGraph*
gs_graph_create(const gsid id)
{
  GSGraph *graph;
  graph = (GSGraph*) malloc(sizeof(GSGraph));
  
  GS_OBJECT(graph)->type = GRAPH_TYPE;

  gs_element_init(GS_ELEMENT(graph),
		  id);

  gs_stream_source_init(GS_SOURCE(graph),
			id);

  gs_stream_sink_init(GS_SINK(graph),
		      graph,
		      GS_SINK_CALLBACK(_gs_graph_sink_callback));

  graph->nodes = g_hash_table_new_full(g_str_hash, g_str_equal, NULL, _gs_graph_node_destroy);
  graph->edges = g_hash_table_new_full(g_str_hash, g_str_equal, NULL, _gs_graph_edge_destroy);

  return graph;
}

GSAPI void
gs_graph_destroy(GSGraph *graph)
{
  gs_element_finalize(GS_ELEMENT(graph));
  gs_stream_source_finalize(GS_SOURCE(graph));
  gs_stream_sink_finalize(GS_SINK(graph));

  g_hash_table_destroy(graph->edges);
  g_hash_table_destroy(graph->nodes);

  free(graph);
}

GSAPI GSNode*
gs_graph_node_add(GSGraph   *graph,
		  const gsid id)
{
  GSNode *node;
  node = g_hash_table_lookup(graph->nodes, id);

  if(node != NULL) {
#ifdef GRAPH_STRICT_CHECKING
    ERROR(GS_ERROR_ID_ALREADY_IN_USE);
#endif
  }
  else {
    gsid nid;
    nid = gs_id_copy(id);
    node = gs_node_create(graph, nid);
    g_hash_table_insert(graph->nodes, nid, node);
  }

  gs_stream_source_trigger_node_added(GS_SOURCE(graph),
				      gs_element_id_get(GS_ELEMENT(graph)),
				      id);

  return node;
}

GSAPI GSNode*
gs_graph_node_get(const GSGraph *graph,
		  const gsid     id)
{
  GSNode *node;
  node = g_hash_table_lookup(graph->nodes, id);

  return node;
}

GSAPI void
gs_graph_node_delete(const GSGraph *graph,
		     const gsid     id)
{
  gs_stream_source_trigger_node_deleted(GS_SOURCE(graph),
				      gs_element_id_get(GS_ELEMENT(graph)),
				      id);

  g_hash_table_remove(graph->nodes, id);
}

GSAPI inline int
gs_graph_node_get_count(const GSGraph *graph)
{
  return g_hash_table_size(graph->nodes);
}

GSAPI GSEdge*
gs_graph_edge_add(GSGraph   *graph,
		  const gsid id,
		  const gsid id_src,
		  const gsid id_trg,
		  gsboolean  directed)
{
  GSEdge *edge;
  edge = g_hash_table_lookup(graph->edges, id);

  if(edge != NULL) {
#ifdef GRAPH_STRICT_CHECKING
    ERROR(GS_ERROR_ID_ALREADY_IN_USE);
#endif
  }
  else {
    GSNode *src, *trg;
    gsid eid;

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
    g_hash_table_insert(graph->edges, eid, edge);

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

GSAPI GSEdge*
gs_graph_edge_get(const GSGraph *graph,
		  const gsid id)
{
  GSEdge *edge;
  edge = g_hash_table_lookup(graph->edges, id);
  
  return edge;
}

GSAPI void
gs_graph_edge_delete(const GSGraph *graph,
		     const gsid id)
{

  gs_stream_source_trigger_edge_deleted(GS_SOURCE(graph),
					gs_element_id_get(GS_ELEMENT(graph)),
					id);

  g_hash_table_remove(graph->edges, id);
}

GSAPI inline int
gs_graph_edge_get_count(const GSGraph *graph)
{
  return g_hash_table_size(graph->edges);
}

GSAPI GSIterator*
gs_graph_node_iterator_new(const GSGraph *graph)
{
  GList *nodes;
  nodes = g_hash_table_get_values(graph->nodes);

  return gs_iterator_list_new(nodes);
}

GSAPI inline void
gs_graph_node_foreach(const GSGraph  *graph,
		      const GSNodeCB  callback,
		      void           *data)
{
  g_hash_table_foreach(graph->nodes, (GHFunc) callback, data);
}

GSAPI GSIterator *
gs_graph_edge_iterator_new(const GSGraph *graph)
{
  GList *edges;
  edges = g_hash_table_get_values(graph->edges);

  return gs_iterator_list_new(edges);
}

GSAPI inline void
gs_graph_edge_foreach(const GSGraph *graph,
		      const GSEdgeCB callback,
		      void          *data)
{
  g_hash_table_foreach(graph->edges, (GHFunc) callback, data);
}
