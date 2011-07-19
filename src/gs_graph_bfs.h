#ifndef _GS_GRAPH_BFS_H_
#define _GS_GRAPH_BFS_H_

#include "gs_types.h"
#include "gs_common.h"
#include "gs_graph.h"

#define GRAPH_BFS_ITERATOR_MAGIC 0xad01f

GSAPI iterator_t*
gs_graph_iterator_bfs_new(const graph_t *graph);

GSAPI iterator_t*
gs_graph_iterator_bfs_new_from_root(const graph_t *graph,
				    const node_t *root);

GSAPI iterator_t*
gs_graph_iterator_bfs_new_from_root_id(const graph_t *graph,
				       element_id_t root);

GSAPI void
gs_graph_iterator_bfs_reset_from_root(iterator_t   *iterator,
				      const node_t *root);

GSAPI unsigned int
gs_graph_iterator_bfs_max_depth(const iterator_t *iterator);

#endif /* _GS_GRAPH_BFS_H_ */
