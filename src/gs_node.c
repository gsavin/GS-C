#include "gs_node.h"

/**********************************************************************
 * PRIVATE
 */



/**********************************************************************
 * PUBLIC
 */

GSAPI GSNode*
gs_node_create(GSGraph   *graph,
	       const gsid id)
{
  GSNode *node;
  node = (GSNode*) malloc(sizeof(GSNode));
  
  GS_OBJECT(node)->type = NODE_TYPE;
  gs_element_init(GS_ELEMENT(node),id);
  node->graph = graph;
  node->edges = NULL;
  node->edge_count = 0;

#ifdef DEBUG
  g_debug("\"%s\"", id);
#endif

  return node;
}

GSAPI void
gs_node_destroy(GSNode *node)
{
#ifdef DEBUG
  g_debug("\"%s\"", gs_element_id_get(GS_ELEMENT(node)));
#endif

  gs_element_finalize(GS_ELEMENT(node));
  g_list_free(node->edges);
  free(node);
}

GSAPI void
gs_node_edge_register(GSNode *node,
		      GSEdge *edge)
{
  if (edge->source == node ||
      edge->target == node) {
    if (g_list_find(node->edges, edge) == NULL)
      node->edges = g_list_append(node->edges, edge);
  }
}

GSAPI inline GSIterator*
gs_node_edge_iterator_new(const GSNode *node)
{
  return gs_iterator_list_new(node->edges);
}
