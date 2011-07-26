#include "gs_stream.h"

/**********************************************************************
 * PRIVATE
 */

GSAPI static inline void
_gs_stream_source_trigger(const GSSource *source,
			  event_t         event,
			  size_t          size,
			  void          **data)
{
  GList  *list;
  GSSink *sink;

  for (list = source->sinks; list; list = list->next) {
    sink = (GSSink*) list->data;
    sink->callback(sink, event, size, data);
  }
}

/**********************************************************************
 * PUBLIC
 */

GSAPI void
gs_stream_source_init(GSSource *source,
		      gsid      id)
{
  source->id = id;
  source->sink_count = 0;
  source->sinks = NULL;
}

GSAPI void
gs_stream_source_finalize(GSSource *source)
{
  g_list_free(source->sinks);
}

GSAPI void
gs_stream_source_sink_add(GSSource     *source,
			  const GSSink *sink)
{
  source->sinks = g_list_append(source->sinks, sink);
  source->sink_count = g_list_length(source->sinks);
}

GSAPI void
gs_stream_source_sink_delete(GSSource     *source,
			     const GSSink *sink)
{
  source->sinks = g_list_remove(source->sinks, sink);
  source->sink_count = g_list_length(source->sinks);
}

GSAPI void
gs_stream_source_trigger_node_added(GSSource *source,
				    gsid      graph_id,
				    gsid      node_id)
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
gs_stream_source_trigger_node_deleted(GSSource *source,
				      gsid      graph_id,
				      gsid      node_id)
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
gs_stream_source_trigger_edge_added(GSSource *source,
				    gsid      graph_id,
				    gsid      edge_id,
				    gsid      edge_source_id,
				    gsid      edge_target_id,
				    gsboolean directed)
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
gs_stream_source_trigger_edge_deleted(GSSource *source,
				      gsid      graph_id,
				      gsid      edge_id)
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
gs_stream_sink_init(GSSink  *sink,
		    void    *container,
		    GSSinkCB callback)
{
  sink->container = container;
  sink->callback  = callback;
}

GSAPI void
gs_stream_sink_finalize(GSSink *sink)
{
  
}
