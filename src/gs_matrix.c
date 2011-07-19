#include "gs_matrix.h"

#define GS_MATRIX_ID_HASH_FUNCTION eina_hash_string_djb2_new
#define SQR(i) (i * i)

/**********************************************************************
 * PRIVATE
 */

typedef struct _node_index node_index;
typedef struct _edge_index edge_index;

struct _node_index {
  unsigned int index;
};

struct _edge_index {
  element_id_t id;
  node_index  *source;
  node_index  *target;
};

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
_matrix_hash_node_free_cb(node_index *data)
{
  free(data);
}

GSAPI static void
_matrix_hash_edge_free_cb(edge_index *data)
{
  gs_id_release(data->id);
  free(data);
}

GSAPI static inline void
_check_size(matrix_t *matrix)
{
  if(matrix->size < SQR(matrix->nodes) * sizeof(GS_MATRIX_FLAG)) {
    EINA_LOG_DBG("not enought memory, make realloc");

    matrix->epr  = (matrix->nodes / GS_MATRIX_ALLOC_STEP + 1) * GS_MATRIX_ALLOC_STEP;
    matrix->size = SQR(matrix->epr) * sizeof(GS_MATRIX_FLAG);
    matrix->data = (GS_MATRIX_FLAG*) realloc(matrix->data, matrix->size);

    EINA_LOG_DBG("matrix size : %dbytes", matrix->size);
  }
}

GSAPI static inline node_index*
_node_index_get(const matrix_t *matrix,
		element_id_t id)
{
  return eina_hash_find(matrix->node_id2index, id);
}

GSAPI static inline edge_index*
_edge_index_get(const matrix_t *matrix,
		element_id_t id)
{
  return eina_hash_find(matrix->edge_id2index, id);
}

#define EDGE_INDEX(m,s,t) ((s * m->epr + t) * sizeof(GS_MATRIX_FLAG))

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

  free(matrix->data);
  free(matrix);
}

GSAPI int
gs_matrix_node_index_get(const matrix_t *m,
			 element_id_t id)
{
  node_index *idx;
  idx = eina_hash_find(m->node_id2index, id);

  if (idx != NULL)
    return idx->index;

  return -1;
}

GSAPI void
gs_matrix_node_add(matrix_t    *matrix,
		   element_id_t id)
{
  int i;
  node_index *index;
  element_id_t nid;

  index = (node_index*) malloc(sizeof(node_index));

  index->index = matrix->nodes;
  matrix->nodes += 1;
  _check_size(matrix);

  nid = gs_id_copy(id);
  eina_hash_add(matrix->node_id2index, nid, index);
  matrix->node_ids = eina_list_append(matrix->node_ids, nid);

  for (i = 0; i <= index->index; i++) {
    matrix->data [EDGE_INDEX(matrix, i, index->index)] = 0;
    matrix->data [EDGE_INDEX(matrix, index->index, i)] = 0;
  }
}

GSAPI void
gs_matrix_edge_add(matrix_t    *matrix,
		   element_id_t id,
		   element_id_t src,
		   element_id_t trg,
		   bool_t       directed)
{
  edge_index *index;
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

  index = eina_hash_find(matrix->edge_id2index, id);
  
  if (index != NULL)
    ERROR(GS_ERROR_ID_ALREADY_IN_USE);

  index = (edge_index*) malloc(sizeof(edge_index));
  index->id     = gs_id_copy(id);
  index->source = _node_index_get(matrix, src);
  index->target = _node_index_get(matrix, trg);

  eina_hash_add(matrix->edge_id2index, id, index);
  matrix->data [EDGE_INDEX(matrix, s, t)] = 1;

  if (!directed)
    matrix->data [EDGE_INDEX(matrix, t, s)] = 1;
}


GSAPI void
gs_matrix_edge_weight_set(matrix_t    *matrix,
			  element_id_t id,
			  real_t       weight)
{
  edge_index *index;
  index = eina_hash_find(matrix->edge_id2index, id);

  if (index == NULL)
    ERROR(GS_ERROR_EDGE_NOT_FOUND);

  
}

GSAPI real_t
gs_matrix_edge_weight_get(const matrix_t *matrix,
			  int source,
			  int target)
{
  return (real_t) matrix->data [EDGE_INDEX(matrix, source, target)];
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
      fprintf(out, "%d ", matrix->data [EDGE_INDEX(matrix, i, j)]);
    fprintf(out, "\n");
    fflush(out);
  }    
}
