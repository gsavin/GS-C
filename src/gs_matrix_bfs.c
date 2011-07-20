#include "gs_matrix_bfs.h"


/**********************************************************************
 * PRIVATE
 */

typedef struct _matrix_iterator_bfs matrix_iterator_bfs;

struct _matrix_iterator_bfs {
  Eina_Iterator parent;

  matrix_t      *matrix;
  int            looking;
  int            candidat;
  int           *stack;
  int           *closed;
  int            depth_max;

  EINA_MAGIC
};

GSAPI static inline void
_bfs_candidat_add(matrix_iterator_bfs *iterator,
		  int candidat,
		  int depth)
{
  if (iterator->closed [candidat] < 0) {
    iterator->stack [iterator->candidat] = candidat;
    iterator->candidat                  += 1;
    iterator->closed [candidat]          = depth;
    
    if (depth > iterator->depth_max)
      iterator->depth_max = depth;
  }
}

GSAPI static inline Eina_Bool
_bfs_next(matrix_iterator_bfs *iterator,
	  void               **data)
{
  return EINA_FALSE;
}

GSAPI static void*
_bfs_get_container(matrix_iterator_bfs *iterator)
{
  return iterator->matrix;
}

GSAPI static void
_bfs_free(matrix_iterator_bfs *iterator)
{
  //matrix_point *data;
  //
  //EINA_LIST_FREE(iterator->stack, data)
  //  free(data);

  free(iterator->stack);
  free(iterator->closed);
  free(iterator);
}

GSAPI static inline matrix_iterator_bfs *
_gs_matrix_iterator_bfs_create(const matrix_t *matrix)
{
  int i;
  matrix_iterator_bfs *iterator;
  iterator = (matrix_iterator_bfs*) malloc(sizeof(matrix_iterator_bfs));
  
  EINA_MAGIC_SET(iterator,          MATRIX_BFS_ITERATOR_MAGIC);
  EINA_MAGIC_SET(&iterator->parent, EINA_MAGIC_ITERATOR);

  iterator->parent.next          = FUNC_ITERATOR_NEXT(_bfs_next);
  iterator->parent.get_container = FUNC_ITERATOR_GET_CONTAINER(_bfs_get_container);
  iterator->parent.free          = FUNC_ITERATOR_FREE(_bfs_free);
  iterator->parent.lock          = FUNC_ITERATOR_LOCK(NULL);
  iterator->parent.unlock        = FUNC_ITERATOR_LOCK(NULL);

  iterator->matrix    = matrix;
  iterator->closed    = (int*) malloc(matrix->nodes * sizeof(int));
  iterator->depth_max = 0;
  iterator->stack     = (int*) malloc(matrix->nodes * sizeof(int));
  iterator->looking   = 0;
  iterator->candidat  = 0;

  for (i = 0; i < matrix->nodes; i++)
    iterator->closed [i] = -1;

  return iterator;
}

/**********************************************************************
 * PUBLIC
 */

GSAPI iterator_t*
gs_matrix_iterator_bfs_new_from_index(const matrix_t *matrix,
				      int index)
{
  matrix_iterator_bfs *iterator;

  iterator = _gs_matrix_iterator_bfs_create(matrix);
  _bfs_candidat_add(iterator, index, 0);

  return (iterator_t*) iterator;
}

GSAPI void
gs_matrix_iterator_bfs_reset_from_index(iterator_t *iterator,
					int index)
{
  if (iterator) {
    matrix_iterator_bfs *bfs;
    int i;
    
    CHECK_MATRIX_BFS_ITERATOR_MAGIC((matrix_iterator_bfs*) iterator, -1);
    
    bfs            = (matrix_iterator_bfs*) iterator;
    bfs->candidat  = 0;
    bfs->depth_max = 0;
    bfs->looking   = 0;

    for (i = 0; i < bfs->matrix->nodes; i++) {
      bfs->stack  [i] = -1;
      bfs->closed [i] = -1;
    }

    _bfs_candidat_add(bfs, index, 0);
  }
}

GSAPI inline int
gs_matrix_iterator_bfs_index_next(iterator_t *iterator)
{
  int index [1];

  CHECK_MATRIX_BFS_ITERATOR_MAGIC((matrix_iterator_bfs*) iterator, -1);

  if (eina_iterator_next(iterator, (void**) &index) == EINA_FALSE)
    return -1;

  return index[0];
}

GSAPI int
gs_matrix_unweighted_eccentricity(matrix_t *matrix,
				  int index)
{
  int  d, dmax;
  int  i, c, o, n, k, nodes;
  int  looking, candidat;
  int *cells, *neigh;
  int *degrees;
  int stack  [matrix->nodes];
  int closed [matrix->nodes];

  cells   = matrix->cells;
  degrees = matrix->degrees;
  dmax    = 0;
  nodes   = matrix->nodes;
  
  k = index;

  if (k < 0)
    k = 0;

 eccentricity:
  for (i = 0; i < nodes; i++)
    closed [i] = -1;
  
  closed [k] = 0;
  stack  [0] = k;
  
  looking  = 0;
  candidat = 1;
  
  while (looking < candidat) {
    n = stack   [looking++];
    d = closed  [n] + 1;
    c = degrees [n];
    
    neigh = cells + n * matrix->davg;
    
    for (i = 0; i < c; i++) {
      o = *(neigh++);
      
      if (closed [o] < 0) {
	stack  [candidat++] = o;
	closed [o]          = d;
      }
    }
  }
  
  d = closed [stack[looking - 1]];
  
  if (d > dmax)
    dmax = d;
  
  if (index < 0 && ++k < nodes)
    goto eccentricity;
  
  return dmax;
}

GSAPI int
gs_matrix_iterator_bfs_depth_max_get(iterator_t *iterator)
{
  CHECK_MATRIX_BFS_ITERATOR_MAGIC((matrix_iterator_bfs*) iterator, -1);
  return ((matrix_iterator_bfs*) iterator)->depth_max;
}

