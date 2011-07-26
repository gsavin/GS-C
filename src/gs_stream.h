#ifndef _GS_STREAM_H_
#define _GS_STREAM_H_

#include "gs_types.h"
#include "gs_common.h"

#ifdef __cplusplus
extern "C" {
#endif

struct _source {
  gsid         id;
  unsigned int sink_count;
  GList       *sinks;
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


typedef void (*GSSinkCB)(const GSSink *sink, const event_t e, size_t size, const void **data);

#define GS_SINK_CALLBACK(function) ((GSSinkCB)function)

struct _sink {
  void *container;
  GSSinkCB callback;
};

#define GS_SOURCE_FIELD GSSource __source
#define GS_SOURCE(e) ((GSSource*) &(e->__source))

#define GS_SINK_FIELD GSSink __sink
#define GS_SINK(e) ((GSSink*) &(e->__sink))

GSAPI void gs_stream_source_init(GSSource *source,
				 gsid      id);

GSAPI void gs_stream_source_finalize(GSSource *source);

GSAPI void gs_stream_source_sink_add(GSSource     *source,
				     const GSSink *sink);

GSAPI void gs_stream_source_sink_delete(GSSource     *source,
					const GSSink *sink);

GSAPI void gs_stream_source_trigger_node_added(GSSource *source,
					       gsid      graph_id,
					       gsid      node_id);

GSAPI void gs_stream_source_trigger_node_deleted(GSSource *source,
						 gsid      graph_id,
						 gsid      node_id);

GSAPI void gs_stream_source_trigger_edge_added(GSSource *source,
					       gsid      graph_id,
					       gsid      edge_id,
					       gsid      edge_source_id,
					       gsid      edge_target_id,
					       gsboolean directed);

GSAPI void gs_stream_source_trigger_edge_deleted(GSSource *source,
						 gsid      graph_id,
						 gsid      edge_id);

GSAPI void gs_stream_sink_init(GSSink  *sink,
			       void    *container,
			       GSSinkCB callback);

GSAPI void gs_stream_sink_finalize(GSSink *sink);

#ifdef __cplusplus
}
#endif

#endif /* _GS_STREAM_H_ */
