#include "gs_matrix_bfs.h"


/**********************************************************************
 * PRIVATE
 */

typedef struct _matrix_iterator_bfs GSMatrixIteratorBFS;

struct _matrix_iterator_bfs {
  GSIterator parent;

  GSMatrix  *matrix;
  int        looking;
  int        candidat;
  int       *stack;
  int       *closed;
  int        depth_max;
};

GSAPI static inline void
_bfs_candidat_add(GSMatrixIteratorBFS *iterator,
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

GSAPI static inline gsboolean
_bfs_next(GSMatrixIteratorBFS *iterator,
	  void               **data)
{
  return GS_FALSE;
}

GSAPI static void
_bfs_free(GSMatrixIteratorBFS *iterator)
{
  //matrix_point *data;
  //
  //EINA_LIST_FREE(iterator->stack, data)
  //  free(data);

  free(iterator->stack);
  free(iterator->closed);
  free(iterator);
}

GSAPI static inline GSMatrixIteratorBFS *
_gs_matrix_iterator_bfs_create(const GSMatrix *matrix)
{
  int i;
  GSMatrixIteratorBFS *iterator;
  iterator = (GSMatrixIteratorBFS*) malloc(sizeof(GSMatrixIteratorBFS));

  iterator->parent.__next = (GSIteratorNextCB) _bfs_next;
  iterator->parent.__free = (GSIteratorFreeCB) _bfs_free;

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

GSAPI GSIterator*
gs_matrix_iterator_bfs_new_from_index(const GSMatrix *matrix,
				      int index)
{
  GSMatrixIteratorBFS *iterator;

  iterator = _gs_matrix_iterator_bfs_create(matrix);
  _bfs_candidat_add(iterator, index, 0);

  return (GSIterator*) iterator;
}

GSAPI void
gs_matrix_iterator_bfs_reset_from_index(GSIterator *iterator,
					int index)
{
  if (iterator) {
    GSMatrixIteratorBFS *bfs;
    int i;
    
    bfs            = (GSMatrixIteratorBFS*) iterator;
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
gs_matrix_iterator_bfs_index_next(GSIterator *iterator)
{
  int index [1];

  //if (gs_iterator_next(iterator, (void**) &index) == GS_FALSE)
    return -1;

    //return index[0];
}

GSAPI int
gs_matrix_unweighted_eccentricity(GSMatrix *matrix,
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
gs_matrix_iterator_bfs_depth_max_get(GSIterator *iterator)
{
  return ((GSMatrixIteratorBFS*) iterator)->depth_max;
}

