#include "gs_common.h"

GSAPI int
gs_init()
{
  int r;

  r = eina_init();
  if(!r)
    return r;

  eina_log_print_cb_set(eina_log_print_cb_stderr, NULL);
#ifdef DEBUG
  eina_log_level_set(EINA_LOG_LEVEL_DBG);
#endif

  return 1;
}

GSAPI void
gs_shutdown()
{
  eina_shutdown();
}

GSAPI inline void
gs_iterator_free(iterator_t *it)
{
  eina_iterator_free(it);
}

GSAPI inline node_t*
gs_iterator_next_node(const iterator_t *it)
{
  node_t *next;

  if(eina_iterator_next(it, (void**) &next) != EINA_TRUE)
    return NULL;

  return next;
}

GSAPI inline edge_t*
gs_iterator_next_edge(const iterator_t *it)
{
  edge_t *next;

  if(eina_iterator_next(it, (void**) &next) != EINA_TRUE)
    return NULL;

  return next;
}
