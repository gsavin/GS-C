#include "gs_matrix.h"
#include <string.h>

#define SQR(i) (i * i)

/**********************************************************************
 * PRIVATE
 */

GSAPI static void
_gs_matrix_sink_callback(const GSSink *sink,
			 event_t       event,
			 size_t        size,
			 const void  **data)
{
  GSMatrix *m;
  m = (GSMatrix*) sink->container;

  switch(event) {
  case NODE_ADDED:
    gs_matrix_node_add(m, (gsid) data[1]);
    break;
  case NODE_DELETED:
    g_warning("Deleting node is not yet implemented in matrix.");
    break;
  case EDGE_ADDED:
    gs_matrix_edge_add(m,
		       (gsid)      data[1],
		       (gsid)      data[2],
		       (gsid)      data[3],
		       (gsboolean) GPOINTER_TO_INT(data[4]));
    break;
  case EDGE_DELETED:
    g_warning("Deleting edge is not yet implemented in matrix.");
    break;
  default:
    break;
  }
}

GSAPI static void
_matrix_hash_edge_free_cb(GSMatrixCell *data)
{
  gs_id_release(data->id);
  free(data);
}

GSAPI static inline void
_check_size(GSMatrix *matrix)
{
  if(matrix->epr <= matrix->nodes || matrix->davg <= matrix->degree_max) {
    int i, e, d, s, *cells;

    // g_debug("not enought memory, make realloc");

    e = matrix->epr;
    d = matrix->davg;
    s = matrix->size;
    i = matrix->epr * matrix->davg;

    if(matrix->epr <= matrix->nodes) {
      matrix->epr  = matrix->nodes / GS_MATRIX_ROW_ALLOC_STEP + 1;
      matrix->epr *= GS_MATRIX_ROW_ALLOC_STEP;
    }

    if (matrix->davg <= matrix->degree_max) {
      matrix->davg  = matrix->degree_max / GS_MATRIX_COLUMN_ALLOC_STEP + 1;
      matrix->davg *= GS_MATRIX_COLUMN_ALLOC_STEP;
    }

    matrix->size    = matrix->epr * matrix->davg * sizeof(int);
    matrix->degrees = (int*) realloc(matrix->degrees, matrix->epr * sizeof(int));
    memset(matrix->degrees + e, 0, (matrix->epr - e) * sizeof(int));

    cells = (int*) malloc(matrix->size);
    memset(cells + i, 0, matrix->size - s);


    for (i = 0; i < e; i++)
      memcpy(cells + i * matrix->davg, matrix->cells + i * d, d * sizeof(int));

    free(matrix->cells);
    matrix->cells = cells;

    // g_debug("matrix size : %d bytes", matrix->size);
  }
}

GSAPI static inline GSMatrixCell*
_cell_get(const GSMatrix *matrix,
	  gsid            id)
{
  return g_hash_table_lookup(matrix->edge_id2index, id);
}

#define EDGE_INDEX(m,s,t) ((s * m->epr + t))

/**********************************************************************
 * PUBLIC
 */

GSAPI GSMatrix*
gs_matrix_new()
{
  GSMatrix *m;
  m = (GSMatrix*) malloc(sizeof(GSMatrix));

  m->epr           = 0;
  m->davg          = 0;
  m->degree_max    = 0;
  m->size          = 0;
  m->nodes         = 0;
  m->edges         = 0;
  m->cells         = NULL;
  m->weights       = NULL;
  m->degrees       = NULL;
  m->node_ids      = NULL;

  m->node_id2index = g_hash_table_new_full(g_str_hash,
					   g_str_equal,
					   NULL,
					   NULL);

  m->edge_id2index = g_hash_table_new_full(g_str_hash,
					   g_str_equal,
					   NULL,
					   (GDestroyNotify) _matrix_hash_edge_free_cb);

  gs_stream_sink_init(GS_SINK(m), m, GS_SINK_CALLBACK(_gs_matrix_sink_callback));

  g_debug("matrix created");

  return m;
}

GSAPI void
gs_matrix_destroy(GSMatrix *matrix)
{
  gsid id;

  g_hash_table_destroy(matrix->node_id2index);
  g_hash_table_destroy(matrix->edge_id2index);
  
  g_list_free_full(matrix->node_ids, (GDestroyNotify) gs_id_release);
  matrix->node_ids = NULL;

  free(matrix->cells);
  free(matrix->degrees);
  free(matrix);
}

GSAPI inline int
gs_matrix_node_index_get(const GSMatrix *matrix,
			 gsid            id)
{
  return GPOINTER_TO_INT(g_hash_table_lookup(matrix->node_id2index, id));
}

GSAPI void
gs_matrix_node_add(GSMatrix *matrix,
		   gsid      id)
{
  int i;
  int index;
  gsid nid;

  index = matrix->nodes;

  matrix->nodes += 1;
  _check_size(matrix);

  matrix->degrees [index] = 0;

  nid = gs_id_copy(id);
  g_hash_table_insert(matrix->node_id2index, nid, GINT_TO_POINTER(index));

  matrix->node_ids = g_list_append(matrix->node_ids, nid);
}

GSAPI void
gs_matrix_edge_add(GSMatrix *matrix,
		   gsid      id,
		   gsid      src,
		   gsid      trg,
		   gsboolean directed)
{
  GSMatrixCell *cell;
  int s, t;

  s = gs_matrix_node_index_get(matrix, src);
  t = gs_matrix_node_index_get(matrix, trg);

  if (s < 0 || t < 0) {
#ifndef GS_AUTOCREATE
    ERROR(GS_ERROR_NODE_NOT_FOUND);
#else
    if (s < 0)
      gs_matrix_node_add(matrix, src);
    
    if (t < 0)
      gs_matrix_node_add(matrix, trg);
#endif
  }

  cell = _cell_get(matrix, id);
  
  if (cell != NULL)
    ERROR(GS_ERROR_ID_ALREADY_IN_USE);

#ifdef GS_MATRIX_HUGE
  if (matrix->data [EDGE_INDEX(matrix, s, t)] != NULL)
    ERROR(GS_ERROR_ID_ALREADY_IN_USE);
#endif
  
  cell         = (GSMatrixCell*) malloc(sizeof(GSMatrixCell));
  cell->id     = gs_id_copy(id);
  cell->source = s;
  cell->target = t;
  
  g_hash_table_insert(matrix->edge_id2index, id, cell);
  
  matrix->edges       += 1;
  matrix->degrees [s] += 1;
  
  if (matrix->degrees [s] > matrix->degree_max)
    matrix->degree_max = matrix->degrees [s];

  if (directed == GS_FALSE) {
    matrix->degrees [t] += 1;

    if (matrix->degrees [s] > matrix->degree_max)
      matrix->degree_max = matrix->degrees [s];
  }

  _check_size(matrix);

  matrix->cells   [N_INDEX(matrix, s, matrix->degrees [s] - 1)] = t;
  //matrix->weights [N_INDEX(matrix, s, matrix->degrees [s] - 1)] = 1;

  if (directed == GS_FALSE) {
    matrix->cells   [N_INDEX(matrix, t, matrix->degrees [t] - 1)] = s;
    //matrix->weights [N_INDEX(matrix, t, matrix->degrees [t] - 1)] = 1;
  }
}


GSAPI inline void
gs_matrix_edge_weight_set(GSMatrix    *matrix,
			  gsid id,
			  gsreal       weight)
{
  
}

GSAPI inline gsreal
gs_matrix_edge_weight_get(const GSMatrix *matrix,
			  int            source,
			  int            target)
{
  GSMatrixCell *cell;

  return 0;
}

GSAPI gsid
gs_matrix_node_id_get(const GSMatrix *matrix,
		      int             index)
{
  return (gsid) g_list_nth_data(matrix->node_ids, index);
}

GSAPI void
gs_matrix_print(const GSMatrix *matrix,
		FILE           *out)
{
  
}

