#ifndef _GS_MATRIX_BFS_H_
#define _GS_MATRIX_BFS_H_

#include "gs_types.h"
#include "gs_common.h"
#include "gs_matrix.h"

#define MATRIX_BFS_ITERATOR_MAGIC 0x0AD0E78

GSAPI GSIterator*
gs_matrix_iterator_bfs_new_from_index(const GSMatrix *matrix,
				      int             index);


GSAPI void
gs_matrix_iterator_bfs_reset_from_index(GSIterator *iterator,
					int         index);

GSAPI int
gs_matrix_iterator_bfs_index_next(GSIterator *iterator);


GSAPI int
gs_matrix_unweighted_eccentricity(GSMatrix *matrix,
				  int       index);

GSAPI int
gs_matrix_iterator_bfs_depth_max_get(GSIterator *iterator);

#endif /* _GS_MATRIX_BFS_H_ */
