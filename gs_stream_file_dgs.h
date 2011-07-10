#ifndef _GS_STREAM_FILE_DGS_H_
#define _GS_STREAM_FILE_DGS_H_

#include "gs_types.h"
#include "gs_common.h"
#include "gs_stream.h"

typedef struct _source_dgs source_dgs;

struct _source_dgs {
  FILE *file;
  GS_SOURCE_FIELD;
};

GSAPI source_dgs *gs_stream_source_file_dgs_open(const char *filename);
GSAPI void gs_stream_source_file_dgs_close(source_dgs *source);
GSAPI bool_t gs_stream_source_file_dgs_next(const source_dgs *source);

#endif /* _GS_STREAM_FILE_DGS_H_ */
