#include "gs_matrix_bfs.h"


/**********************************************************************
 * PRIVATE
 */

typedef struct _matrix_point matrix_point;
typedef struct _matrix_iterator_bfs matrix_iterator_bfs;

struct _matrix_point {
  int row;
  int column;
};

typedef enum {
  NOT_VISITED = 0,
  VISITED     = 1,
  EXPLORED    = 2
} closed_flag;

struct _matrix_iterator_bfs {
  Eina_Iterator parent;

  matrix_t      *matrix;
  //int            row;
  //int            column;
  //Eina_List     *stack;
  int            looking;
  int            candidat;
  int           *stack;
  int           *closed;
  unsigned int   depth_max;

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

GSAPI static Eina_Bool
_bfs_next(matrix_iterator_bfs *iterator,
	  void               **data)
{
  if (iterator->looking < iterator->candidat) {
    int i, n;
    matrix_t *m;

    m = iterator->matrix;
    n = iterator->stack [iterator->looking];
    
    *data = n;

    iterator->looking += 1;

    for (i = 0; i < m->nodes; i++)
      if (gs_matrix_edge_weight_get(m, n, i) > 0)
	_bfs_candidat_add(iterator, i, iterator->closed [n] + 1);
    
    return EINA_TRUE;
  }

  return EINA_FALSE;
}

/*
GSAPI static Eina_Bool
_bfs_next_old(matrix_iterator_bfs *iterator,
	  void               **data)
{
  int i;
  int n;
  matrix_point *parent;

  if (iterator->row < 0)
    return EINA_FALSE;

  if (iterator->row >= 0 && iterator->column < 0) {
    EINA_LOG_DBG("use row");
    
    *data = (void*) iterator->row;
    iterator->column = 0;

    //if (iterator->row < iterator->matrix->nodes)
    //  iterator->closed [iterator->row] = VISITED;
  }
  else if (iterator->row >= 0) {
    EINA_LOG_DBG("use column");

    *data = (void*) iterator->column;

    //if( iterator->column < iterator->matrix->nodes)
    //  iterator->closed [iterator->column] = VISITED;
  }
  else
    return EINA_FALSE;

  n = -1;

  while (n < 0) {
    for (i = iterator->column; i < iterator->matrix->nodes && n < 0; i++) {
      if (gs_matrix_edge_weight_get(iterator->matrix, iterator->row, i) > 0 &&
	  iterator->closed [i] == NOT_VISITED)
	n = i;
    }

    if (n < 0) {
      EINA_LOG_DBG("no more column");

      for (i = 0; i < iterator->matrix->nodes && n < 0; i++) {
	if (gs_matrix_edge_weight_get(iterator->matrix, iterator->row, i) > 0 &&
	    iterator->closed [i] != EXPLORED)
	  n = i;
      }
      
      if (n < 0) {
	if (iterator->stack != NULL) {
	  matrix_point *p;
	  p = eina_list_data_get(iterator->stack);
	  
	  iterator->row    = p->row;
	  iterator->column = p->column;
	  iterator->stack  = eina_list_remove_list(iterator->stack, iterator->stack);
	  
	  EINA_LOG_DBG("pop parent");

	  free(p);
	}
	else {
	  iterator->row    = -1;
	  iterator->column = -1;
	  
	  n = iterator->matrix->nodes;
	}
      }
      else {
	parent           = (matrix_point*) malloc(sizeof(matrix_point));
	parent->row      = iterator->row;
	parent->column   = iterator->column;
	iterator->stack  = eina_list_prepend(iterator->stack, parent);
	iterator->row    = n;
	iterator->column = 0;

	EINA_LOG_DBG("parent : %d;%d -> %d;%d", parent->row, parent->column, n, 0);

	iterator->closed [n] = EXPLORED;

	n = -1;

	if (eina_list_count(iterator->stack) > iterator->depth_max)
	  iterator->depth_max = eina_list_count(iterator->stack);
      }
    }
    else {
      EINA_LOG_DBG("next column (%d)", n);
      iterator->closed [n] = VISITED;
      iterator->column = n;
    }
  }

  return EINA_TRUE;
}
*/

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

GSAPI static void
_bfs_lock(matrix_iterator_bfs *iterator)
{
  // TODO
}

GSAPI static void
_bfs_unlock(matrix_iterator_bfs *iterator)
{
  // TODO
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
  iterator->parent.lock          = FUNC_ITERATOR_LOCK(_bfs_lock);
  iterator->parent.unlock        = FUNC_ITERATOR_LOCK(_bfs_unlock);

  iterator->matrix    = matrix;
  iterator->closed    = (closed_flag*) malloc(matrix->nodes * sizeof(closed_flag));
  iterator->depth_max = 0;
  iterator->stack     = (int*) malloc(matrix->nodes * sizeof(int));//NULL;
  iterator->looking   = 0;
  iterator->candidat  = 0;
  //iterator->row       = -1;
  //iterator->column    = -1;

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

  //iterator->row    = index;
  //iterator->column = -1;
  //iterator->closed [index] = EXPLORED;
  _bfs_candidat_add(iterator, index, 0);

  return (iterator_t*) iterator;
}

GSAPI int
gs_matrix_iterator_bfs_index_next(iterator_t *iterator)
{
  int index [1];

  CHECK_MATRIX_BFS_ITERATOR_MAGIC((matrix_iterator_bfs*) iterator, -1);

  if (eina_iterator_next(iterator, (void**) &index) == EINA_FALSE)
    return -1;

  return index[0];
}

GSAPI int
gs_matrix_iterator_bfs_depth_max_get(iterator_t *iterator)
{
  CHECK_MATRIX_BFS_ITERATOR_MAGIC((matrix_iterator_bfs*) iterator, -1);
  return ((matrix_iterator_bfs*) iterator)->depth_max;
}

