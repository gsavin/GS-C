#define STACK(i) stack[i]
#define CLOSE(i) close[i]

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
    int   *neigh, *stack;
    float  d;
    float *close;

    close = (float*) malloc(nodes*sizeof(float));
    stack = (int*) malloc(nodes*sizeof(int));

    for (i = 0; i < nodes; i++)
      CLOSE(i) = -1;
    
    CLOSE(index) = 0;
    STACK(0)     = index;
    
    looking  = 0;
    candidat = 1;
    
    while (looking < candidat) {
      n = stack   [looking++];
      d = CLOSE(n) + 1;
      c = degrees [n];
      
      neigh = cells + n * padding;
      
      for (i = 0; i < c; i++) {
	o = *(neigh++);
	
	if (CLOSE(o) < 0) {
	  STACK(candidat++) = o;
	  CLOSE(o)          = d;
	}
      }
    }

    looking = STACK(looking - 1);
    eccentricities [index] = CLOSE(looking);

    free(stack);
    free(close);
  }
}
