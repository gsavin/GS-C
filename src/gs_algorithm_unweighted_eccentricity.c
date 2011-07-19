#include "gs_algorithm_unweighted_eccentricity.h"
#include "gs_graph_bfs.h"

/**********************************************************************
 * PRIVATE
 */

GSAPI static int
_unweighted_eccentricity(iterator_t *it)
{
  int r;

  if (it != NULL) {
    while (gs_iterator_next_node(it) != NULL)
      ;

    r = gs_graph_iterator_bfs_depth_max(it);
    gs_iterator_free(it);
  }
  else {
    EINA_LOG_DBG("iterator is NULL");
    r = -1;
  }

  return r;
}

/**********************************************************************
 * PUBLIC
 */

GSAPI int
gs_algorithm_unweighted_eccentricity(const graph_t *graph,
				     const node_t *node)
{
  iterator_t *it;
  it = gs_graph_iterator_bfs_new_from_root(graph,
					   node);

  return _unweighted_eccentricity(it);
}

GSAPI int
gs_algorithm_unweighted_eccentricity_max(const graph_t *graph)
{
  iterator_t *it;
  it = gs_graph_iterator_bfs_new(graph);

  return _unweighted_eccentricity(it);
}
