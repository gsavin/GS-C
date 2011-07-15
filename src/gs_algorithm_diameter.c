#include "gs_algorithm_diameter.h"
#include "gs_algorithm_unweighted_eccentricity.h"

GSAPI int
gs_algorithm_diameter(const graph_t *graph)
{
  int         d1, d2;
  iterator_t *nodes;
  node_t     *node;

  d1    = 0;
  nodes = gs_graph_node_iterator_new(graph);
  node  = gs_iterator_next_node(nodes);

  while (node != NULL) {
    //printf("done \"%s\"\n", gs_element_id_get(GS_ELEMENT(node)));

    d2   = gs_algorithm_unweighted_eccentricity(graph, node);
    d1   = d2 > d1 ? d2 : d1;
    node = gs_iterator_next_node(nodes);
  }

  gs_iterator_free(nodes);

  return d1;
}
