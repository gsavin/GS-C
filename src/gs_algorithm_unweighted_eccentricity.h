#ifndef _GS_ALGORITHM_UNWEIGHTED_ECCENTRICITY_H_
#define _GS_ALGORITHM_UNWEIGHTED_ECCENTRICITY_H_

#include "gs_graph.h"

#ifdef __cplusplus
extern "C" {
#endif

  GSAPI int
  gs_algorithm_unweighted_eccentricity(const GSGraph *graph,
				       const GSNode *node);
  
  GSAPI int
  gs_algorithm_unweighted_eccentricity_max(const GSGraph *graph);
  
#ifdef __cplusplus
}
#endif

#endif /* _GS_ALGORITHM_UNWEIGHTED_ECCENTRICITY_H_ */

