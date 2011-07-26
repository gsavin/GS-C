#ifndef _GS_ID_H_
#define _GS_ID_H_

#include "gs_types.h"
#include "gs_common.h"

#ifdef __cplusplus
extern "C" {
#endif

GSAPI gsid gs_id_copy(const gsid id);
GSAPI void gs_id_release(gsid id);

#ifdef __cplusplus
}
#endif

#endif /* _GS_ID_H_ */
