#include <cublas.h>
#include "gs_matrix.h"

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

GSAPI __host__ int
gs_cuda_diameter(const matrix_t *matrix)
{
  float ecc;
  int ind;
  int *degrees_device, *data_device;
  float *ecc_device;
  dim3 block(16);
  dim3 grid(matrix->nodes / 16 + 1);

  cudaMalloc((void**) &degrees_device, matrix->nodes * sizeof(int));
  cudaMalloc((void**) &data_device,    matrix->size);
  cudaMalloc((void**) &ecc_device,     matrix->nodes * sizeof(float));

  cudaMemcpy(degrees_device, matrix->degrees, matrix->nodes * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(data_device,    matrix->cells,    matrix->size,                cudaMemcpyHostToDevice);

  diameter<<<grid, block>>>(matrix->nodes, degrees_device, data_device, matrix->davg, ecc_device);
  ind = cublasIsamax(matrix->nodes, ecc_device, 1);

  cudaMemcpy(&ecc, ecc_device + ind, sizeof(float), cudaMemcpyDeviceToHost);
}
