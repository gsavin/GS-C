#include "gs_stream.h"

/**********************************************************************
 * PRIVATE
 */

GSAPI static inline void
_gs_stream_source_trigger(const source_t *source,
			  event_t event,
			  size_t size,
			  const void **data)
{
  Eina_List *list;
  Eina_List *l;
  Eina_List *l_next;
  sink_t      *sink;

  list = source->sinks;

  EINA_LIST_FOREACH_SAFE(list, l, l_next, data)
    sink->callback(sink, event, size, data);
}

/**********************************************************************
 * PUBLIC
 */

GSAPI void
gs_stream_source_init(source_t *source)
{
  source->sinks = NULL;
}

GSAPI void
gs_stream_source_finalize(source_t *source)
{
  eina_list_free(source->sinks);
}

GSAPI void
gs_stream_source_sink_add(source_t *source,
			  const sink_t *sink)
{
  source->sinks = eina_list_append(source->sinks, sink);

  if(eina_error_get())
    ERROR(GS_ERROR_CAN_NOT_ADD_SINK);

  source->sink_count = eina_list_count(source->sinks);
}

GSAPI void
gs_stream_source_sink_delete(source_t *source,
			     const sink_t *sink)
{
  source->sinks = eina_list_remove(source->sinks, sink);

  if(eina_error_get())
    ERROR(GS_ERROR_CAN_NOT_DELETE_SINK);

  source->sink_count = eina_list_count(source->sinks);
}

GSAPI void
gs_stream_source_trigger_node_added(source_t *source,
				    element_id_t graph_id,
				    element_id_t node_id)
{
  void **data;

  if(source->sink_count) {
    data = (void**) malloc(2*sizeof(void*));
    data[0] = graph_id;
    data[1] = node_id;
    
    _gs_stream_source_trigger(source, NODE_ADDED, 2, data);
    
    free(data);
  }
}

GSAPI void
gs_stream_source_trigger_node_deleted(source_t *source,
				      element_id_t graph_id,
				      element_id_t node_id)
{
  void **data;

  if(source->sink_count) {
    data = (void**) malloc(2*sizeof(void*));
    data[0] = graph_id;
    data[1] = node_id;
    
    _gs_stream_source_trigger(source, NODE_DELETED, 2, data);
    
    free(data);
  }
}

GSAPI void
gs_stream_source_trigger_edge_added(source_t *source,
				    element_id_t graph_id,
				    element_id_t edge_id,
				    element_id_t edge_source_id,
				    element_id_t edge_target_id,
				    bool_t directed)
{
  void **data;

  if(source->sink_count) {
    data = (void**) malloc(5*sizeof(void*));
    data[0] = graph_id;
    data[1] = edge_id;
    data[2] = edge_source_id;
    data[3] = edge_target_id;
    data[4] = directed;
    
    _gs_stream_source_trigger(source, EDGE_ADDED, 5, data);
    
    free(data);
  }
}

GSAPI void
gs_stream_source_trigger_edge_deleted(source_t *source,
				      element_id_t graph_id,
				      element_id_t edge_id)
{
  void **data;

  if(source->sink_count) {
    data = (void**) malloc(2*sizeof(void*));
    data[0] = graph_id;
    data[1] = edge_id;
    
    _gs_stream_source_trigger(source, EDGE_DELETED, 2, data);
    
    free(data);
  }
}

GSAPI void
gs_stream_sink_init(sink_t *sink, sink_cb_t callback)
{
  sink->callback = callback;
}

GSAPI void
gs_stream_sink_finalize(sink_t *sink)
{
  
}
