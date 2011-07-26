#ifndef _GS_STREAM_DGS_H_
#define _GS_STREAM_DGS_H_

#include "gs_types.h"
#include "gs_common.h"
#include "gs_stream.h"

#ifdef __cplusplus
extern "C" {
#endif

  typedef struct _source_dgs GSSourceDGS;
  typedef struct _sink_dgs GSSinkDGS;

  struct _source_dgs {
    GIOChannel *in;
    GS_SOURCE_FIELD;
  };

  struct _sink_dgs {
    GIOChannel *out;
    GS_SINK_FIELD;
  };

#define DGS_SINK(sink) ((GSSinkDGS*)(sink->container))

  GSAPI GSSourceDGS*
  gs_stream_source_file_dgs_open(const char *filename);

  GSAPI void
  gs_stream_source_file_dgs_close(GSSourceDGS *source);

  GSAPI gsboolean
  gs_stream_source_file_dgs_next(const GSSourceDGS *source);

  GSAPI GSSinkDGS*
  gs_stream_sink_file_dgs_open(const char *filename);

  GSAPI void
  gs_stream_sink_file_dgs_close(GSSinkDGS *sink);

#ifdef __cplusplus
}
#endif

#endif /* _GS_STREAM_DGS_H_ */
