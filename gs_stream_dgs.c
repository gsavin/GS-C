#include <string.h>
#include "gs_stream_dgs.h"

/**********************************************************************
 * PRIVATE
 */

typedef enum {
  DGS_AN, DGS_CN, DGS_DN,
  DGS_AE, DGS_CE, DGS_DE,
  DGS_CG, DGS_ST
} dgs_event_t;

GSAPI static inline void
_eat_end_of_line(FILE *in,
		 bool_t spaces_allowed,
		 bool_t all_allowed)
{
  char c;
  
  while( ((c = fgetc(in)) == ' ' || c == '\t' || (c != '\n' && all_allowed))
	 && (spaces_allowed || all_allowed))
    ;

  if(c != '\n')
    ERROR(GS_ERROR_IO);
}

GSAPI static inline void
_eat_spaces(FILE *in)
{
  int c;
  
  while((c = fgetc(in)) == ' '|| c == '\t')
    ;

  ungetc(c, in);
}

GSAPI static void
_gs_stream_source_dgs_read_header(const source_dgs_t *source)
{
  char *magic;
  size_t r;

  magic = (char*) malloc(6*sizeof(char));
  r = fread(magic, 6, sizeof(char), source->in);

  if(r<6)
    ERROR(GS_ERROR_IO);

  if(strcmp(magic,"DGS004")) {
    EINA_LOG_WARN("Invalid DGS magic header");
    ERROR(GS_ERROR_IO);
  }

  free(magic);

  _eat_end_of_line(source->in, GS_FALSE, GS_FALSE);
  
  // Next line is deprecated but still present.
  // Skipping it.
  _eat_end_of_line(source->in, GS_TRUE, GS_TRUE);
}

GSAPI static inline dgs_event_t
_gs_stream_source_dgs_read_event_type(const source_dgs_t *source)
{
  char c1, c2;

  c1 = fgetc(source->in);
  c2 = fgetc(source->in);

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
  default:
    ERROR(GS_ERROR_IO);
  }
}

GSAPI static inline element_id_t
_dgs_read_id(FILE *in)
{
  Eina_Strbuf *buffer;
  char *id;
  char c;
  bool_t backslash;

  buffer = eina_strbuf_new();

  c = fgetc(in);

  if(c == '"') {
    while( (c = fgetc(in)) != '"' || backslash) {
      if(c == '\\') {
	backslash = GS_TRUE;
      }
      else {
	if(!eina_strbuf_append_char(buffer, c))
	  ERROR(GS_ERROR_BUFFER_ERROR);
	
	backslash = GS_FALSE;
      }
    }
  }
  else {
    while( (c >= 'a' && c <= 'z') ||
	   (c >= 'A' && c <= 'Z') ||
	   (c >= '0' && c <= '9') ||
	   c == '_' || c == '-' || c == '.' ) {
      
	if(!eina_strbuf_append_char(buffer, c))
	  ERROR(GS_ERROR_BUFFER_ERROR);

	c = fgetc(in);
    }

    if(c != ' ' && c != '\t' && c != '\n' && c != EOF)
      ERROR(GS_ERROR_INVALID_ID_FORMAT);

    ungetc(c, in);
  }

  id = eina_strbuf_string_steal(buffer);
  eina_strbuf_free(buffer);

  return id;
}

GSAPI static inline void
_dgs_read_event_an(const source_dgs_t *source)
{
  element_id_t id;
  id = _dgs_read_id(source->in);

  // TODO : Handle attributes
  _eat_end_of_line(source->in, GS_TRUE, GS_TRUE);

  gs_stream_source_trigger_node_added(GS_SOURCE(source),
				      GS_SOURCE(source)->id,
				      id);

  free(id);
}

GSAPI static inline void
_dgs_read_event_cn(const source_dgs_t *source)
{
  // TODO
  _eat_end_of_line(source->in, GS_TRUE, GS_TRUE);
}

GSAPI static inline void
_dgs_read_event_dn(const source_dgs_t *source)
{
  element_id_t id;
  id = _dgs_read_id(source->in);
  _eat_end_of_line(source->in, GS_TRUE, GS_FALSE);

  gs_stream_source_trigger_node_deleted(GS_SOURCE(source),
					GS_SOURCE(source)->id,
					id);

  free(id);
}

GSAPI static inline void
_dgs_read_event_ae(const source_dgs_t *source)
{
  element_id_t id;
  element_id_t node_source, node_target;
  int c;
  bool_t directed;

  directed = GS_FALSE;

  id = _dgs_read_id(source->in);
  _eat_spaces(source->in);
  node_source = _dgs_read_id(source->in);
  _eat_spaces(source->in);

  c = fgetc(source->in);

  switch(c) {
  case '<':
    directed = GS_TRUE;
    _eat_spaces(source->in);
    node_target = node_source;
    node_source = _dgs_read_id(source->in);
    break;
  case '>':
    directed = GS_TRUE;
    _eat_spaces(source->in);
    node_target = _dgs_read_id(source->in);
    break;
  default:
    ungetc(c, source->in);
    node_target = _dgs_read_id(source->in);
    break;
  }
  
  gs_stream_source_trigger_edge_added(GS_SOURCE(source),
				      GS_SOURCE(source)->id,
				      id,
				      node_source,
				      node_target,
				      directed);
  
  // TODO : Handle attributes
  _eat_end_of_line(source->in, GS_TRUE, GS_TRUE);

  free(id);
  free(node_source);
  free(node_target);
}

GSAPI static inline void
_dgs_read_event_ce(const source_dgs_t *source)
{
  // TODO
  _eat_end_of_line(source->in, GS_TRUE, GS_TRUE);
}

GSAPI static inline void
_dgs_read_event_de(const source_dgs_t *source)
{
  element_id_t id;
  id = _dgs_read_id(source->in);

  gs_stream_source_trigger_edge_deleted(GS_SOURCE(source),
					GS_SOURCE(source)->id,
					id);

  free(id);
}

GSAPI static inline void
_dgs_read_event_cg(const source_dgs_t *source)
{
  // TODO
  _eat_end_of_line(source->in, GS_TRUE, GS_TRUE);
}

GSAPI static inline void
_dgs_read_event_st(const source_dgs_t *source)
{
  // TODO
  _eat_end_of_line(source->in, GS_TRUE, GS_TRUE);
}

/**********************************************************************
 * PUBLIC
 */

GSAPI source_dgs_t*
gs_stream_source_file_dgs_open(const char *filename)
{
  source_dgs_t *source;
  FILE *in;

  in = fopen(filename, "r");

  if(in) {
    source = (source_dgs_t*) malloc(sizeof(source_dgs_t));
    gs_stream_source_init(GS_SOURCE(source), filename);
    source->in = in;

    return source;
  }
  
  return NULL;
}

GSAPI void
gs_stream_source_file_dgs_close(source_dgs_t *source)
{
  if(!source)
    return;

  fclose(source->in);
  gs_stream_source_finalize(GS_SOURCE(source));
  
  free(source);
}

GSAPI bool_t
gs_stream_source_file_dgs_next(const source_dgs_t *source)
{
  dgs_event_t e;
  e = _gs_stream_source_dgs_read_event_type(source);

  switch(e) {
  case DGS_AN:
    _dgs_read_event_an(source);
    break;
  case DGS_CN:
    _dgs_read_event_cn(source);
    break;
  case DGS_DN:
    _dgs_read_event_dn(source);
    break;
  case DGS_AE:
    _dgs_read_event_ae(source);
    break;
  case DGS_CE:
    _dgs_read_event_ce(source);
    break;
  case DGS_DE:
    _dgs_read_event_de(source);
    break;
  case DGS_CG:
    _dgs_read_event_cg(source);
    break;
  case DGS_ST:
    _dgs_read_event_st(source);
    break;
  }
}
