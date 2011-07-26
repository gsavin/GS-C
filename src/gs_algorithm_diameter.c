#include "gs_algorithm_diameter.h"
#include "gs_algorithm_unweighted_eccentricity.h"

GSAPI int
gs_algorithm_diameter(const GSGraph *graph)
{
  int         d1, d2;
  GSIterator *nodes;
  GSNode     *node;

  d1    = 0;
  nodes = gs_graph_node_iterator_new(graph);
  node  = gs_iterator_next_node(nodes);

  while (node != NULL) {
    d2   = gs_algorithm_unweighted_eccentricity(graph, node);
    d1   = d2 > d1 ? d2 : d1;
    node = gs_iterator_next_node(nodes);
  }

  gs_iterator_free(nodes);

  return d1;
}
