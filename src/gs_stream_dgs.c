#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "gs_stream_dgs.h"

/**********************************************************************
 * PRIVATE
 */

typedef enum {
  DGS_AN, DGS_CN, DGS_DN,
  DGS_AE, DGS_CE, DGS_DE,
  DGS_CG, DGS_ST, DGS_EOF
} dgs_event_t;

GSAPI static inline void
_handle_io_status(GIOStatus status, GError **error)
{
  switch(status) {
  case G_IO_STATUS_ERROR:
    ERROR(GS_ERROR_IO);
    break;
  case G_IO_STATUS_EOF:
    break;
  case G_IO_STATUS_AGAIN:
    ERROR(GS_ERROR_IO);
    break;
  case G_IO_STATUS_NORMAL:
    break;
  }

  if (*error != NULL) {
    g_error_free(*error);
    *error = NULL;
  }
}

GSAPI static inline gunichar
_read_char(GIOChannel *in)
{
  GIOStatus status;
  GError   *err;
  gunichar  the_char;
  
  status = g_io_channel_read_unichar(in, &the_char, &err);
  _handle_io_status(status, &err);

  return the_char;
}

GSAPI static inline void
_eat_end_of_line(GIOChannel *in,
		 gsboolean   spaces_allowed,
		 gsboolean   all_allowed)
{
  GIOStatus status;
  GError   *err;
  gchar    *buffer;
  gsize     length;
  gsize     term;
  gsize     i;

  status = g_io_channel_read_line(in, &buffer, &length, &term, &err);
  _handle_io_status(status, &err);
  
  for (i = 0; i < term; i++) {
    if (!all_allowed &&
	!(spaces_allowed && (buffer[i] == ' ' || buffer[i] == '\t')))
      ERROR(GS_ERROR_IO);
  }

  if (buffer)
    g_free(buffer);
}

GSAPI static inline void
_eat_spaces(const GString *buffer,
	    int           *from)
{
  while(buffer->str[*from] == ' ' || buffer->str[*from] == '\t')
    (*from)++;
}

GSAPI static void
_gs_stream_source_dgs_read_header(const GSSourceDGS *source)
{
  GIOStatus status;
  GError   *err;
  GString  *buffer;
  gsize     term;

  err    = NULL;
  buffer = g_string_new(NULL);

  status = g_io_channel_read_line_string(source->in, buffer, &term, &err);
  _handle_io_status(status, &err);

  if (term != 6 || !g_str_has_prefix(buffer->str, "DGS004")) {
    g_error("Invalid DGS magic header");
    ERROR(GS_ERROR_IO);
  }
  
  // Next line is deprecated but still present.
  // Skipping it.
  status = g_io_channel_read_line_string(source->in, buffer, &term, &err);
  _handle_io_status(status, &err);

  g_string_free(buffer, TRUE);
}

GSAPI static inline dgs_event_t
_gs_stream_source_dgs_read_event_type(const GString *buffer,
				      int           *pos)
{
  gchar c1, c2;

  c1 = buffer->str [(*pos)++];
  c2 = buffer->str [(*pos)++];

  if(c1 >= 'A' && c1 <= 'Z')
    c1 = c1 - ('A' - 'a');

  if(c2 >= 'A' && c2 <= 'Z')
    c2 = c2 - ('A' - 'a');

  switch(c2) {
  case 'n':
    switch(c1) {
    case 'a':
      return DGS_AN;
    case 'c':
      return DGS_CN;
    case 'd':
      return DGS_DN;
    default:
      ERROR(GS_ERROR_IO);
    }
  case 'e':
    switch(c1) {
    case 'a':
      return DGS_AE;
    case 'c':
      return DGS_CE;
    case 'd':
      return DGS_DE;
    default:
      ERROR(GS_ERROR_IO);
    }
  case 'g':
    if(c1 != 'c')
      ERROR(GS_ERROR_IO);

    return DGS_CG;
  case 't':
    if(c1 != 's')
      ERROR(GS_ERROR_IO);
    
    return DGS_ST;
  case EOF:
    return DGS_EOF;
  default:
    ERROR(GS_ERROR_IO);
  }
}

GSAPI static inline gchar*
_dgs_read_id(const GString *buffer,
	     int           *pos)
{
  GString  *id;
  gchar     c;
  gsboolean backslash;

  _eat_spaces(buffer, pos);

  c  = buffer->str[(*pos)++];
  id = g_string_new(NULL);

  if(c == '"') {
    while( (c = buffer->str[(*pos)++]) != '"' || backslash) {
      if(c == '\\') {
	backslash = GS_TRUE;
      }
      else {
	id = g_string_append_c(id, c);
	backslash = GS_FALSE;
      }
    }
  }
  else {
    while( (c >= 'a' && c <= 'z') ||
	   (c >= 'A' && c <= 'Z') ||
	   (c >= '0' && c <= '9') ||
	   c == '_' || c == '-' || c == '.' ) {
      
	id = g_string_append_c(id, c);
	c = buffer->str[(*pos)++];
    }

    if(c != ' ' && c != '\t' && c != '\n' && c != EOF)
      ERROR(GS_ERROR_INVALID_ID_FORMAT);

    *pos--;
  }

  return g_string_free(id, FALSE);
}

GSAPI static inline void
_dgs_read_event_an(const GSSource *source,
		   const GString  *buffer,
		   int            *pos)
{
  gchar *id;
  id = _dgs_read_id(buffer, pos);

  // TODO : Handle attributes

  gs_stream_source_trigger_node_added(source, source->id, id);
  g_free(id);
}

GSAPI static inline void
_dgs_read_event_cn(const GSSource *source,
		   const GString  *buffer,
		   int            *pos)
{
  // TODO
}

GSAPI static inline void
_dgs_read_event_dn(const GSSource *source,
		   const GString  *buffer,
		   int            *pos)
{
  gchar *id;
  id = _dgs_read_id(buffer, pos);

  gs_stream_source_trigger_node_deleted(source, source->id, id);
  g_free(id);
}

GSAPI static inline void
_dgs_read_event_ae(const GSSource *source,
		   const GString  *buffer,
		   int            *pos)
{
  gchar     *id;
  gchar     *node_source;
  gchar     *node_target;
  gchar      c;
  gsboolean  directed;

  directed = GS_FALSE;

  id = _dgs_read_id(buffer, pos);
  _eat_spaces(buffer, pos);
  node_source = _dgs_read_id(buffer, pos);
  _eat_spaces(buffer, pos);

  c = buffer->str [(*pos)++];

  switch(c) {
  case '<':
    directed = GS_TRUE;
    _eat_spaces(buffer, pos);
    node_target = node_source;
    node_source = _dgs_read_id(buffer, pos);
    break;
  case '>':
    directed = GS_TRUE;
    _eat_spaces(buffer, pos);
    node_target = _dgs_read_id(buffer, pos);
    break;
  default:
    (*pos)--;
    node_target = _dgs_read_id(buffer, pos);
    break;
  }

  gs_stream_source_trigger_edge_added(source,
				      source->id,
				      id,
				      node_source,
				      node_target,
				      directed);
  
  // TODO : Handle attributes

  g_free(id);
  g_free(node_source);
  g_free(node_target);
}

GSAPI static inline void
_dgs_read_event_ce(const GSSource *source,
		   const GString  *buffer,
		   int            *pos)
{
  // TODO
}

GSAPI static inline void
_dgs_read_event_de(const GSSource *source,
		   const GString  *buffer,
		   int            *pos)
{
  gchar *id;
  id = _dgs_read_id(buffer, pos);

  gs_stream_source_trigger_edge_deleted(source,
					source->id,
					id);

  g_free(id);
}

GSAPI static inline void
_dgs_read_event_cg(const GSSource *source,
		   const GString  *buffer,
		   int            *pos)
{
  // TODO
}

GSAPI static inline void
_dgs_read_event_st(const GSSource *source,
		   const GString  *buffer,
		   int            *pos)
{
  // TODO
}

GSAPI static void
_dgs_sink_callback(const GSSink *sink,
		   const event_t e,
		   size_t        size,
		   const void  **data)
{
  GSSinkDGS *dgs;
  GString   *buffer;
  GIOStatus  status;
  GError    *err;
  gsize      bytes;

  err = NULL;

  dgs    = DGS_SINK(sink);
  buffer = g_string_new("");

  switch(e) {
  case NODE_ADDED:
    assert(size > 1 );
    g_string_append_printf(buffer, "an \"%s\"\n", data[1]);
    break;
  case NODE_DELETED:
    assert(size > 1 );
    g_string_append_printf(buffer, "dn \"%s\"\n", data[1]);
    break;
  case EDGE_ADDED:
    assert(size > 4 );

    if((gsboolean) data[4])
       g_string_append_printf(buffer, "ae \"%s\" \"%s\" > \"%s\"\n",
			      data[1],
			      data[2],
			      data[3]);
    else
       g_string_append_printf(buffer, "ae \"%s\" \"%s\" \"%s\"\n",
			      data[1],
			      data[2],
			      data[3]);
    break;
  case EDGE_DELETED:
    assert(size > 1 );
    g_string_append_printf(buffer, "de \"%s\"\n", data[1]);
    break;
  }

  g_string_append_c(buffer, '\0');

  status = g_io_channel_write_chars(dgs->out, buffer->str, -1, &bytes, &err);
  _handle_io_status(status, &err);

  g_string_free(buffer, TRUE);
}

/**********************************************************************
 * PUBLIC
 */

GSAPI GSSourceDGS*
gs_stream_source_file_dgs_open(const char *filename)
{
  GSSourceDGS *source;
  GIOChannel  *in;
  GError      *err;

  err = NULL;

  in = g_io_channel_new_file(filename, "r", &err);

  if (err != NULL)
    g_error_free(err);

  if(in) {
    source = (GSSourceDGS*) malloc(sizeof(GSSourceDGS));
    gs_stream_source_init(GS_SOURCE(source), filename);
    source->in = in;
    
    _gs_stream_source_dgs_read_header(source);

    return source;
  }
  
  // TODO : Handle err
  return NULL;
}

GSAPI void
gs_stream_source_file_dgs_close(GSSourceDGS *source)
{
  GIOStatus status;
  GError   *err;

  if(!source)
    return;

  err = NULL;

  status = g_io_channel_shutdown(source->in, FALSE, &err);
  _handle_io_status(status, &err);
  g_io_channel_unref(source->in);
  
  gs_stream_source_finalize(GS_SOURCE(source));
  
  free(source);
}

GSAPI gsboolean
gs_stream_source_file_dgs_next(const GSSourceDGS *source)
{
  dgs_event_t e;
  GString    *line;
  int         pos;
  GIOStatus   status;
  GError     *err;
  gsize       term;
  
  err  = NULL;
  line = g_string_new(NULL);
  term = 0;

  do {
    status = g_io_channel_read_line_string(source->in, line, &term, &err);
    _handle_io_status(status, &err);
  } while(status != G_IO_STATUS_EOF && term == 0);

  pos    = 0;
  
  if (status == G_IO_STATUS_EOF)
    e = DGS_EOF;
  else
    e = _gs_stream_source_dgs_read_event_type(line, &pos);

  switch(e) {
  case DGS_AN:
    _dgs_read_event_an(GS_SOURCE(source), line, &pos);
    break;
  case DGS_CN:
    _dgs_read_event_cn(GS_SOURCE(source), line, &pos);
    break;
  case DGS_DN:
    _dgs_read_event_dn(GS_SOURCE(source), line, &pos);
    break;
  case DGS_AE:
    _dgs_read_event_ae(GS_SOURCE(source), line, &pos);
    break;
  case DGS_CE:
    _dgs_read_event_ce(GS_SOURCE(source), line, &pos);
    break;
  case DGS_DE:
    _dgs_read_event_de(GS_SOURCE(source), line, &pos);
    break;
  case DGS_CG:
    _dgs_read_event_cg(GS_SOURCE(source), line, &pos);
    break;
  case DGS_ST:
    _dgs_read_event_st(GS_SOURCE(source), line, &pos);
    break;
  case DGS_EOF:
    g_string_free(line, TRUE);
    return GS_FALSE;
  }

  g_string_free(line, TRUE);

  return GS_TRUE;
}

GSAPI GSSinkDGS*
gs_stream_sink_file_dgs_open(const char *filename)
{
  GSSinkDGS  *dgs;
  GIOChannel *out;
  GError     *err;
  GIOStatus   status;
  gsize       size;

  err = NULL;
  out = g_io_channel_new_file(filename, "w", &err);
  
  if (err != NULL)
    g_error_free(err);

  if(out == NULL)
    ERROR(GS_ERROR_IO);

  dgs = (GSSinkDGS*) malloc(sizeof(GSSinkDGS));
  dgs->out = out;
  
  gs_stream_sink_init(GS_SINK(dgs),
		      dgs,
		      GS_SINK_CALLBACK(_dgs_sink_callback));

  status = g_io_channel_write_chars(out, "DGS004\nnull 0 0\n\0", -1, &size, &err);
  _handle_io_status(status, &err);
  status = g_io_channel_flush(out, &err);
  _handle_io_status(status, &err);

  return dgs;
}

GSAPI void
gs_stream_sink_file_dgs_close(GSSinkDGS *sink)
{
  GError   *err;
  GIOStatus status;

  err = NULL;

  status = g_io_channel_shutdown(sink->out, TRUE, &err);
  _handle_io_status(status, &err);
  g_io_channel_unref(sink->out);

  sink->out = NULL;
  gs_stream_sink_finalize(GS_SINK(sink));

  free(sink);
}
