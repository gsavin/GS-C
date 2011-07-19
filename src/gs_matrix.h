#ifndef _GS_MATRIX_H_
#define _GS_MATRIX_H_

#include "gs_types.h"
#include "gs_common.h"
#include "gs_stream.h"

#ifndef GS_MATRIX_WEIGHTED
# define GS_MATRIX_FLAG unsigned char
#else
# define GS_MATRIX_FLAG real_t
#endif

#ifndef GS_MATRIX_ALLOC_STEP
# define GS_MATRIX_ALLOC_STEP 128
#endif

#ifndef GS_MATRIX_ROW_CELLS_ALLOC
#define GS_MATRIX_ROW_CELLS_ALLOC 8
#endif

typedef unsigned int        matrix_index_t;
typedef struct _matrix      matrix_t;
typedef struct _matrix_row  matrix_row;
typedef struct _matrix_cell matrix_cell;

struct _matrix {
  matrix_cell    **data;
  size_t           epr;
  size_t           size;
  unsigned int     nodes;
  unsigned int     edges;
  Eina_Hash       *node_id2index;
  Eina_List       *node_ids;
  Eina_Hash       *edge_id2index;
  matrix_row     **rows;
  GS_SINK_FIELD;
  EINA_MAGIC
};

struct _matrix_row {
  unsigned int  index;
  int           degree;
  int           size;
  matrix_cell **cells;
};

struct _matrix_cell {
  element_id_t id;
  matrix_row  *source;
  matrix_row  *target;
  real_t       weight;
};

#define GS_MATRIX_MAGIC 0x0AD0E

GSAPI matrix_t *gs_matrix_new();
GSAPI void gs_matrix_destroy(matrix_t *matrix);

GSAPI void gs_matrix_node_add(matrix_t    *matrix,
			      element_id_t id);

GSAPI void gs_matrix_edge_add(matrix_t    *matrix,
			      element_id_t id,
			      element_id_t src,
			      element_id_t trg,
			      bool_t       directed);

GSAPI real_t gs_matrix_edge_weight_get(const matrix_t *matrix,
				       int source,
				       int target);

GSAPI element_id_t gs_matrix_node_id_get(const matrix_t *matrix,
					 int index);

GSAPI void gs_matrix_print(const matrix_t *matrix,
			   FILE *out);

GSAPI inline int
gs_matrix_row_cell_count(const matrix_t *matrix,
			 int index);

GSAPI inline matrix_cell**
gs_matrix_row_cells_get(const matrix_t *matrix,
			     int index);

GSAPI inline int
gs_matrix_row_cell_nth_index(const matrix_t *matrix,
			     int index,
			     int neighbor);

#endif /* _GS_MATRIX_H_ */
