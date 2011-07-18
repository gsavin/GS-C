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

struct _matrix_iterator_bfs {
  Eina_Iterator parent;

  matrix_t     *matrix;
  int           row;
  int           column;
  matrix_point  next;
  Eina_List    *stack;
  bool_t       *closed;
};

GSAPI static Eina_Bool
_bfs_next(matrix_iterator_bfs *iterator,
	  void               **data)
{
  int i;
  int n;

  if (iterator->row < 0)
    return EINA_FALSE;

  *data = iterator->row;

  n = -1;

  for (i = iterator->column >= 0 ? iterator->column : 0; i < iterator->matrix->nodes && n < 0; i++) {
    if (gs_matrix_edge_weight_get(iterator->matrix, iterator->row, i) > 0 &&
	iterator->closed [i] > 0)
      n = i;
  }

  if (n < 0) {

  }
  else {
    iterator->stack  = eina_list_prepend(iterator->stack, iterator->row);
    iterator->row    = n;
    iterator->column = -1;
  }
}

/**********************************************************************
 * PUBLIC
 */

GSAPI iterator_t*
gs_matrix_bfs_new(const matrix_t *matrix)
{
  
}


