#include <cublas.h>
#include "gs_cuda_diameter.h"

__global__ void
diameter(int    nodes,
	 int   *degrees,
	 int   *cells,
	 int    padding,
	 float *eccentricities);

GSAPI int
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

  return ecc;
}
