#include "gs_matrix.h"

#define GS_MATRIX_ID_HASH_FUNCTION eina_hash_string_djb2_new
#define SQR(i) (i * i)

/**********************************************************************
 * PRIVATE
 */

GSAPI static void
_gs_matrix_sink_callback(const sink_t *sink,
			 event_t event,
			 size_t size,
			 const void **data)
{
  matrix_t *m;
  m = (matrix_t*) sink->container;

  switch(event) {
  case NODE_ADDED:
    gs_matrix_node_add(m, (element_id_t) data[1]);
    break;
  case NODE_DELETED:
    EINA_LOG_WARN("Deleting node is not yet implemented in matrix.");
    break;
  case EDGE_ADDED:
    gs_matrix_edge_add(m,
		      (element_id_t) data[1],
		      (element_id_t) data[2],
		      (element_id_t) data[3],
		      (bool_t) data[4]);
    break;
  case EDGE_DELETED:
    EINA_LOG_WARN("Deleting edge is not yet implemented in matrix.");
    break;
  default:
    break;
  }
}

GSAPI static void
_matrix_hash_node_free_cb(int *data)
{
  free(data);
}

GSAPI static void
_matrix_hash_edge_free_cb(matrix_cell *data)
{
  gs_id_release(data->id);
  free(data);
}

GSAPI static inline void
_check_size(matrix_t *matrix)
{
  if(matrix->epr <= matrix->nodes || matrix->davg <= matrix->degree_max) {
    int i, e, d, s, *cells;

    EINA_LOG_DBG("not enought memory, make realloc");

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

    EINA_LOG_DBG("matrix size : %d bytes", matrix->size);
  }
}

GSAPI static inline int*
_node_index_get(const matrix_t *matrix,
	 element_id_t id)
{
  return eina_hash_find(matrix->node_id2index, id);
}

GSAPI static inline matrix_cell*
_cell_get(const matrix_t *matrix,
	  element_id_t id)
{
  return eina_hash_find(matrix->edge_id2index, id);
}

#define EDGE_INDEX(m,s,t) ((s * m->epr + t))

/**********************************************************************
 * PUBLIC
 */

GSAPI matrix_t*
gs_matrix_new()
{
  matrix_t *m;
  m = (matrix_t*) malloc(sizeof(matrix_t));

  EINA_MAGIC_SET(m, GS_MATRIX_MAGIC);

  m->epr           = 0;
  m->davg          = 0;
  m->degree_max    = 0;
  m->size          = 0;
  m->nodes         = 0;
  m->edges         = 0;
  m->cells         = NULL;
  m->weights       = NULL;
  m->degrees       = NULL;
  m->node_id2index = GS_MATRIX_ID_HASH_FUNCTION(EINA_FREE_CB(_matrix_hash_node_free_cb));
  m->node_ids      = NULL;
  m->edge_id2index = GS_MATRIX_ID_HASH_FUNCTION(EINA_FREE_CB(_matrix_hash_edge_free_cb));

  gs_stream_sink_init(GS_SINK(m), m, GS_SINK_CALLBACK(_gs_matrix_sink_callback));

  EINA_LOG_DBG("matrix created");

  return m;
}

GSAPI void
gs_matrix_destroy(matrix_t *matrix)
{
  element_id_t id;

  eina_hash_free(matrix->node_id2index);
  eina_hash_free(matrix->edge_id2index);
  
  EINA_LIST_FREE(matrix->node_ids, id)
    gs_id_release(id);

  free(matrix->cells);
  free(matrix->degrees);
  free(matrix);
}

GSAPI int
gs_matrix_node_index_get(const matrix_t *m,
			 element_id_t id)
{
  int *index;
  index = _node_index_get(m, id);

  if (index != NULL)
    return *index;

  return -1;
}

GSAPI void
gs_matrix_node_add(matrix_t    *matrix,
		   element_id_t id)
{
  int i;
  int *index;
  element_id_t nid;

  index  = (int*) malloc(sizeof(int));
  *index = matrix->nodes;

  matrix->nodes += 1;
  _check_size(matrix);

  matrix->degrees [*index] = 0;

  nid = gs_id_copy(id);
  eina_hash_add(matrix->node_id2index, nid, index);

  matrix->node_ids = eina_list_append(matrix->node_ids, nid);
}

GSAPI void
gs_matrix_edge_add(matrix_t    *matrix,
		   element_id_t id,
		   element_id_t src,
		   element_id_t trg,
		   bool_t       directed)
{
  matrix_cell *cell;
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
  
  cell         = (matrix_cell*) malloc(sizeof(matrix_cell));
  cell->id     = gs_id_copy(id);
  cell->source = s;
  cell->target = t;
  
  eina_hash_add(matrix->edge_id2index, id, cell);
  
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
gs_matrix_edge_weight_set(matrix_t    *matrix,
			  element_id_t id,
			  real_t       weight)
{
  
}

GSAPI inline real_t
gs_matrix_edge_weight_get(const matrix_t *matrix,
			  int source,
			  int target)
{
  matrix_cell *cell;

  return 0;
}

GSAPI element_id_t
gs_matrix_node_id_get(const matrix_t *matrix,
		 int index)
{
  return (element_id_t) eina_list_nth(matrix->node_ids, index);
}

GSAPI void
gs_matrix_print(const matrix_t *matrix,
		FILE *out)
{
  
}

