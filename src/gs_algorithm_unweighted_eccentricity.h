#ifndef _GS_ALGORITHM_UNWEIGHTED_ECCENTRICITY_H_
#define _GS_ALGORITHM_UNWEIGHTED_ECCENTRICITY_H_

#include "gs_graph.h"

GSAPI int
gs_algorithm_unweighted_eccentricity(const graph_t *graph,
				     const node_t *node);

GSAPI int
gs_algorithm_unweighted_eccentricity_max(const graph_t *graph);

#endif /* _GS_ALGORITHM_UNWEIGHTED_ECCENTRICITY_H_ */

