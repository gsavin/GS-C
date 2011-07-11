#ifndef _GS_STREAM_H_
#define _GS_STREAM_H_

#include "gs_types.h"
#include "gs_common.h"

struct _source {
  element_id_t id;
  unsigned int sink_count;
  Eina_List *sinks;
};

typedef enum {
  NODE_ADDED,
  NODE_DELETED,
  EDGE_ADDED,
  EDGE_DELETED,
  GRAPH_ATTRIBUTE_ADDED,
  GRAPH_ATTRIBUTE_CHANGED,
  GRAPH_ATTRIBUTE_DELETED,
  NODE_ATTRIBUTE_ADDED,
  NODE_ATTRIBUTE_CHANGED,
  NODE_ATTRIBUTE_DELETED,
  EDGE_ATTRIBUTE_ADDED,
  EDGE_ATTRIBUTE_CHANGED,
  EDGE_ATTRIBUTE_DELETED
} event_t;


typedef void (*sink_cb_t)(const sink_t *sink, const event_t e, size_t size, const void **data);

#define GS_SINK_CALLBACK(function) ((sink_cb_t)function)

struct _sink {
  sink_cb_t callback;
};

#define GS_SOURCE_FIELD source_t __source
#define GS_SOURCE(e) ((source_t*) &(e->__source))

#define GS_SINK_FIELD sink_t __sink
#define GS_SINK(e) ((sink_t*) &(e->__sink))

GSAPI void gs_stream_source_init(source_t *source,
				 element_id_t id);

GSAPI void gs_stream_source_finalize(source_t *source);

GSAPI void gs_stream_source_sink_add(source_t *source,
				     const sink_t *sink);

GSAPI void gs_stream_source_sink_delete(source_t *source,
					const sink_t *sink);

GSAPI void gs_stream_source_trigger_node_added(source_t *source,
					       element_id_t graph_id,
					       element_id_t node_id);

GSAPI void gs_stream_source_trigger_node_deleted(source_t *source,
						 element_id_t graph_id,
						 element_id_t node_id);

GSAPI void gs_stream_source_trigger_edge_added(source_t *source,
					       element_id_t graph_id,
					       element_id_t edge_id,
					       element_id_t edge_source_id,
					       element_id_t edge_target_id,
					       bool_t directed);

GSAPI void gs_stream_source_trigger_edge_deleted(source_t *source,
						 element_id_t graph_id,
						 element_id_t edge_id);

GSAPI void gs_stream_sink_init(sink_t *sink,
			       sink_cb_t callback);

GSAPI void gs_stream_sink_finalize(sink_t *sink);

#endif /* _GS_STREAM_H_ */
