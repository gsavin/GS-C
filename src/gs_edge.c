#include "gs_edge.h"

GSAPI edge_t *
gs_edge_create(const graph_t *g,
	       const element_id_t id,
	       const node_t *src,
	       const node_t *trg,
	       bool_t directed)
{
  edge_t *edge;
  edge = (edge_t*) malloc(sizeof(edge_t));
  
  GS_OBJECT(edge)->type = EDGE_TYPE;
  gs_element_init(GS_ELEMENT(edge),id);
  
  edge->graph = g;
  edge->source = src;
  edge->target = trg;
  edge->directed = directed;

#ifdef DEBUG
  EINA_LOG_DBG("edge \"%s\" created", id);
#endif

  return edge;
}

GSAPI void
gs_edge_destroy(edge_t *edge)
{
#ifdef DEBUG
  EINA_LOG_DBG("edge \"%s\" destroyed", gs_element_id_get(GS_ELEMENT(edge)));
#endif

  gs_element_finalize(GS_ELEMENT(edge));
  free(edge);
}
