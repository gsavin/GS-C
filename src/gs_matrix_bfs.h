#ifndef _GS_MATRIX_BFS_H_
#define _GS_MATRIX_BFS_H_

#include "gs_types.h"
#include "gs_common.h"
#include "gs_matrix.h"

#define MATRIX_BFS_ITERATOR_MAGIC 0x0AD0E78

#define CHECK_MATRIX_BFS_ITERATOR_MAGIC(d, ...)			\
  do {								\
    if (!EINA_MAGIC_CHECK(d, MATRIX_BFS_ITERATOR_MAGIC)) {	\
      EINA_MAGIC_FAIL(d, MATRIX_BFS_ITERATOR_MAGIC);		\
      return __VA_ARGS__;					\
    }								\
  } while(0)

GSAPI iterator_t*
gs_matrix_iterator_bfs_new_from_index(const matrix_t *matrix,
				      int index);


GSAPI void
gs_matrix_iterator_bfs_reset_from_index(iterator_t *iterator,
					int index);

GSAPI int
gs_matrix_iterator_bfs_index_next(iterator_t *iterator);


GSAPI int
gs_matrix_unweighted_eccentricity(matrix_t *matrix,
				  int       index);

GSAPI int
gs_matrix_iterator_bfs_depth_max_get(iterator_t *iterator);

#endif /* _GS_MATRIX_BFS_H_ */
