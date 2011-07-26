#ifndef _GS_CUDA_DIAMETER_H_
#define _GS_CUDA_DIAMETER_H_

#include "gs_types.h"
#include "gs_common.h"
#include "gs_matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

  extern GSAPI int
  gs_cuda_diameter(const GSMatrix *matrix);

#ifdef __cplusplus
}
#endif

#endif /*  _GS_CUDA_DIAMETER_H_ */
