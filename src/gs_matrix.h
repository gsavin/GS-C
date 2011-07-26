#ifndef _GS_MATRIX_H_
#define _GS_MATRIX_H_

#include "gs_types.h"
#include "gs_common.h"
#include "gs_stream.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef GS_MATRIX_WEIGHTED
# define GS_MATRIX_FLAG unsigned char
#else
# define GS_MATRIX_FLAG real_t
#endif

#ifndef GS_MATRIX_ROW_ALLOC_STEP
#define GS_MATRIX_ROW_ALLOC_STEP 128
#endif

#ifndef GS_MATRIX_COLUMN_ALLOC_STEP
#define GS_MATRIX_COLUMN_ALLOC_STEP 16
#endif

typedef struct _matrix      matrix_t;
typedef struct _matrix_cell matrix_cell;

struct _matrix {
  int             *cells;
  int             *degrees;
  real_t          *weights;
  size_t           epr;
  size_t           davg;
  size_t           size;
  unsigned int     degree_max;
  unsigned int     nodes;
  unsigned int     edges;
  Eina_Hash       *node_id2index;
  Eina_List       *node_ids;
  Eina_Hash       *edge_id2index;

  GS_SINK_FIELD;
  EINA_MAGIC
};

struct _matrix_cell {
  element_id_t id;
  int          source;
  int          target;
};

#define GS_MATRIX_MAGIC 0x0AD0E

#define N_INDEX(m,n,i) (n * m->davg + i)

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

#ifdef __cplusplus
}
#endif

#endif /* _GS_MATRIX_H_ */
