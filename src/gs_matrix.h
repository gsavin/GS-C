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

typedef unsigned int matrix_index_t;
typedef struct _matrix matrix_t;

struct _matrix {
  GS_MATRIX_FLAG  *data;
  size_t           epr;
  size_t           size;
  unsigned int     nodes;
  unsigned int     edges;
  Eina_Hash       *node_id2index;
  Eina_List       *node_ids;
  Eina_Hash       *edge_id2index;
  GS_SINK_FIELD;
  EINA_MAGIC
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

GSAPI void gs_matrix_print(const matrix_t *matrix,
			   FILE *out);

#endif /* _GS_MATRIX_H_ */
