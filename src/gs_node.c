#include "gs_node.h"

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
  EINA_LOG_DBG("node \"%s\" created", id);
#endif

  return node;
}

GSAPI void
gs_node_destroy(node_t *node)
{
#ifdef DEBUG
  EINA_LOG_DBG("node \"%s\" destroyed", gs_element_id_get(GS_ELEMENT(node)));
#endif

  gs_element_finalize(GS_ELEMENT(node));
  eina_list_free(node->edges);
  free(node);
}
