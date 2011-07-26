#include "gs_edge.h"

GSAPI GSEdge *
gs_edge_create(GSGraph    *g,
	       const gsid  id,
	       GSNode     *src,
	       GSNode     *trg,
	       gsboolean   directed)
{
  GSEdge *edge;
  edge = (GSEdge*) malloc(sizeof(GSEdge));
  
  GS_OBJECT(edge)->type = EDGE_TYPE;
  gs_element_init(GS_ELEMENT(edge),id);
  
  edge->graph = g;
  edge->source = src;
  edge->target = trg;
  edge->directed = directed;

#ifdef DEBUG
  g_debug("\"%s\"", id);
#endif

  return edge;
}

GSAPI void
gs_edge_destroy(GSEdge *edge)
{
#ifdef DEBUG
  g_debug("\"%s\"", gs_element_id_get(GS_ELEMENT(edge)));
#endif

  gs_element_finalize(GS_ELEMENT(edge));
  free(edge);
}

