#include "gs_cuda_diameter.h"

__global__ void diameter(const int    nodes,
			 int   *degrees,
			 int   *cells,
                         int    padding,
			 float *eccentricities)
{
  int index;
  index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < nodes) {
    int    looking, candidat, i, o, n, c;
    int   *stack, *neigh;
    float *closed;
    float  d;

    stack = (int*) malloc(nodes*sizeof(int));
    closed = (float*) malloc(nodes*sizeof(float));

    for (i = 0; i < nodes; i++)
      closed [i] = -1;
    
    closed [index] = 0;
    stack  [0]     = index;
    
    looking  = 0;
    candidat = 1;
    
    while (looking < candidat) {
      n = stack   [looking++];
      d = closed  [n] + 1;
      c = degrees [n];
      
      neigh = cells + n * padding;
      
      for (i = 0; i < c; i++) {
	o = *(neigh++);
	
	if (closed [o] < 0) {
	  stack  [candidat++] = o;
	  closed [o]          = d;
	}
      }
    }
    
    eccentricities [index] = closed [stack[looking - 1]];
  }
}
