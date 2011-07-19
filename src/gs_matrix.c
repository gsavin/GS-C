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
_matrix_hash_node_free_cb(matrix_row *data)
{
  free(data->cells);
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
  if(matrix->size < SQR(matrix->nodes) * sizeof(matrix_cell*)) {
    int i;

    EINA_LOG_DBG("not enought memory, make realloc");

    i = SQR(matrix->epr);

    matrix->epr  = (matrix->nodes / GS_MATRIX_ALLOC_STEP + 1) * GS_MATRIX_ALLOC_STEP;
    matrix->size = SQR(matrix->epr) * sizeof(matrix_cell*);
    matrix->data = (matrix_cell**) realloc(matrix->data, matrix->size);
    matrix->rows = (matrix_row**) realloc(matrix->rows, matrix->epr * sizeof(matrix_row*));

    for (; i < SQR(matrix->epr); i++)
      matrix->data [i] = NULL;

    EINA_LOG_DBG("matrix size : %d bytes", matrix->size);
  }
}

GSAPI static inline matrix_row*
_row_get(const matrix_t *matrix,
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

  m->size          = 0;
  m->epr           = 0;
  m->data          = NULL;
  m->nodes         = 0;
  m->edges         = 0;
  m->rows          = NULL;
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

  free(matrix->rows);
  free(matrix->data);
  free(matrix);
}

GSAPI int
gs_matrix_node_index_get(const matrix_t *m,
			 element_id_t id)
{
  matrix_row *row;
  row = _row_get(m, id);

  if (row != NULL)
    return row->index;

  return -1;
}

GSAPI void
gs_matrix_node_add(matrix_t    *matrix,
		   element_id_t id)
{
  int i;
  matrix_row *row;
  element_id_t nid;

  row = (matrix_row*) malloc(sizeof(matrix_row));
  row->index  = matrix->nodes;
  row->cells  = NULL;
  row->degree = 0;
  row->size   = 0;

  matrix->nodes += 1;
  _check_size(matrix);

  nid = gs_id_copy(id);
  eina_hash_add(matrix->node_id2index, nid, row);

  matrix->node_ids          = eina_list_append(matrix->node_ids, nid);
  matrix->rows [row->index] = row;
}

GSAPI static inline void
_check_row_size(matrix_row *row)
{
  if (row->size <= row->degree) {
    row->size += GS_MATRIX_ROW_CELLS_ALLOC;
    row->cells = (matrix_cell**) realloc(row->cells, row->size * sizeof(matrix_cell*));
  }
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

  if (matrix->data [EDGE_INDEX(matrix, s, t)] == NULL) {
    cell         = (matrix_cell*) malloc(sizeof(matrix_cell));
    cell->id     = gs_id_copy(id);
    cell->source = matrix->rows [s];
    cell->target = matrix->rows [t];
    cell->weight = 1.0;
    
    eina_hash_add(matrix->edge_id2index, id, cell);
    
    matrix->data [EDGE_INDEX(matrix, s, t)] = cell;

    cell->source->degree += 1;
    _check_row_size(cell->source);
    cell->source->cells [cell->source->degree - 1] = cell;
  }
  else
    EINA_LOG_WARN("edge already exists between these nodes");

  if (!directed) {
    if (matrix->data [EDGE_INDEX(matrix, t, s)] != NULL)
      ERROR(GS_ERROR_ID_ALREADY_IN_USE);
    
    matrix->data [EDGE_INDEX(matrix, t, s)] = cell;
    
    cell->target->degree += 1;
    _check_row_size(cell->target);
    cell->target->cells [cell->target->degree - 1] = cell;
  }
}


GSAPI inline void
gs_matrix_edge_weight_set(matrix_t    *matrix,
			  element_id_t id,
			  real_t       weight)
{
  matrix_cell *cell;
  cell = _cell_get(matrix, id);

  if (index == NULL)
    ERROR(GS_ERROR_EDGE_NOT_FOUND);

  cell->weight = weight;
}

GSAPI inline real_t
gs_matrix_edge_weight_get(const matrix_t *matrix,
			  int source,
			  int target)
{
  matrix_cell *cell;
  cell =matrix->data [EDGE_INDEX(matrix, source, target)];

  if (cell != NULL)
    return cell->weight;

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
  int i, j;

  for (i = 0; i < matrix->nodes; i++) {
    for (j = 0; j < matrix->nodes; j++)
      fprintf(out, "%.0f ", gs_matrix_edge_weight_get(matrix, i, j));
    fprintf(out, "\n");
    fflush(out);
  }    
}

GSAPI inline iterator_t*
gs_matrix_row_cell_iterator_new(const matrix_t* matrix,
				int index)
{
  return eina_list_iterator_new(matrix->rows [index]->cells);
}

GSAPI inline int
gs_matrix_row_cell_count(const matrix_t *matrix,
			 int index)
{
  return eina_list_count(matrix->rows [index]->cells);
}

GSAPI inline matrix_cell**
gs_matrix_row_cells_get(const matrix_t *matrix,
			     int index)
{
  return matrix->rows [index]->cells;
}

GSAPI inline int
gs_matrix_row_cell_nth_target(const matrix_t *matrix,
			      int index,
			      int neighbor)
{
  matrix_cell *cell;
  cell = (matrix_cell*) eina_list_nth(matrix->rows [index]->cells, neighbor);

  if (cell != NULL)
    return cell->source->index == index ?
      cell->target->index : cell->source->index;

  return -1;
}
