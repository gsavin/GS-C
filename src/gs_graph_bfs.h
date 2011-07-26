#ifndef _GS_GRAPH_BFS_H_
#define _GS_GRAPH_BFS_H_

#include "gs_types.h"
#include "gs_common.h"
#include "gs_graph.h"

#define GRAPH_BFS_ITERATOR_MAGIC 0xad01f

#ifdef __cplusplus
extern "C" {
#endif

GSAPI GSIterator*
gs_graph_iterator_bfs_new(const GSGraph *graph);

GSAPI GSIterator*
gs_graph_iterator_bfs_new_from_root(const GSGraph *graph,
				    const GSNode  *root);

GSAPI GSIterator*
gs_graph_iterator_bfs_new_from_root_id(const GSGraph *graph,
				       gsid           root);

GSAPI void
gs_graph_iterator_bfs_reset_from_root(GSIterator   *iterator,
				      const GSNode *root);

GSAPI unsigned int
gs_graph_iterator_bfs_max_depth(const GSIterator *iterator);

#ifdef __cplusplus
}
#endif

#endif /* _GS_GRAPH_BFS_H_ */
