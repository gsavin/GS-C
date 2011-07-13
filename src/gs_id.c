#include "gs_id.h"
#include <string.h>

GSAPI element_id_t
gs_id_copy(const element_id_t id)
{
  element_id_t clone;
  unsigned int l;

  l = strlen(id);

  clone = (element_id_t) malloc((1+l)*sizeof(char));
  strncpy(clone, id, l);
  clone[l] = '\0';

  return clone;
}

GSAPI void
gs_id_release(element_id_t id)
{
  free(id);
}
