#include "gs_id.h"
#include <string.h>

GSAPI gsid
gs_id_copy(const gsid id)
{
  gsid clone;
  unsigned int l;

  l = strlen(id);

  clone = (gsid) malloc((1+l)*sizeof(char));
  strncpy(clone, id, l);
  clone[l] = '\0';

  return clone;
}

GSAPI void
gs_id_release(gsid id)
{
  free(id);
}
