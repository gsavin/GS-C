#ifndef _GS_ID_H_
#define _GS_ID_H_

#include "gs_types.h"
#include "gs_common.h"

GSAPI element_id_t gs_id_copy(const element_id_t id);
GSAPI void gs_id_release(element_id_t id);

#endif /* _GS_ID_H_ */
