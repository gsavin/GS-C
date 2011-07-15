#include "gs_node.h"

/**********************************************************************
 * PRIVATE
 */



/**********************************************************************
 * PUBLIC
 */

GSAPI node_t *
gs_node_create(const graph_t *g,
	       const element_id_t id)
{
  node_t *node;
  node = (node_t*) malloc(sizeof(node_t));
  
  GS_OBJECT(node)->type = NODE_TYPE;
  gs_element_init(GS_ELEMENT(node),id);
  node->graph = g;
  node->edges = NULL;
  node->edge_count = 0;

#ifdef DEBUG
  EINA_LOG_DBG("\"%s\"", id);
#endif

  return node;
}

GSAPI void
gs_node_destroy(node_t *node)
{
#ifdef DEBUG
  EINA_LOG_DBG("\"%s\"", gs_element_id_get(GS_ELEMENT(node)));
#endif

  gs_element_finalize(GS_ELEMENT(node));
  eina_list_free(node->edges);
  free(node);
}

GSAPI void
gs_node_edge_register(node_t *node,
		      const edge_t *edge)
{
  if (edge->source == node ||
      edge->target == node) {
    if (eina_list_data_find(node->edges, edge) == NULL)
      node->edges = eina_list_append(node->edges, edge);
  }
}

GSAPI iterator_t*
gs_node_edge_iterator_new(const node_t *node)
{
  return eina_list_iterator_new(node->edges);
}
