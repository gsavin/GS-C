#ifndef _GS_MATRIX_H_
#define _GS_MATRIX_H_

#include "gs_types.h"
#include "gs_common.h"
#include "gs_stream.h"
#include "gs_id.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef GS_MATRIX_WEIGHTED
# define GS_MATRIX_FLAG unsigned char
#else
# define GS_MATRIX_FLAG gsreal
#endif

#ifndef GS_MATRIX_ROW_ALLOC_STEP
#define GS_MATRIX_ROW_ALLOC_STEP 128
#endif

#ifndef GS_MATRIX_COLUMN_ALLOC_STEP
#define GS_MATRIX_COLUMN_ALLOC_STEP 16
#endif

typedef struct _matrix      GSMatrix;
typedef struct _matrix_cell GSMatrixCell;

struct _matrix {
  int             *cells;
  int             *degrees;
  gsreal          *weights;
  size_t           epr;
  size_t           davg;
  size_t           size;
  unsigned int     degree_max;
  unsigned int     nodes;
  unsigned int     edges;
  GHashTable      *node_id2index;
  GList           *node_ids;
  GHashTable      *edge_id2index;

  GS_SINK_FIELD;
};

struct _matrix_cell {
  gsid id;
  int          source;
  int          target;
};

#define GS_MATRIX_MAGIC 0x0AD0E

#define N_INDEX(m,n,i) (n * m->davg + i)

GSAPI GSMatrix *gs_matrix_new();
GSAPI void gs_matrix_destroy(GSMatrix *matrix);

GSAPI void gs_matrix_node_add(GSMatrix *matrix,
			      gsid      id);

GSAPI void gs_matrix_edge_add(GSMatrix *matrix,
			      gsid      id,
			      gsid      src,
			      gsid      trg,
			      gsboolean directed);

GSAPI gsreal gs_matrix_edge_weight_get(const GSMatrix *matrix,
				       int             source,
				       int             target);

GSAPI gsid gs_matrix_node_id_get(const GSMatrix *matrix,
				 int             index);

GSAPI void gs_matrix_print(const GSMatrix *matrix,
			   FILE           *out);

#ifdef __cplusplus
}
#endif

#endif /* _GS_MATRIX_H_ */
