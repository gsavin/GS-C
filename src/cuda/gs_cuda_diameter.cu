#include <cublas.h>
#include "gs_cuda_diameter.h"

__global__ void
diameter(int    nodes,
	 int   *degrees,
	 int   *cells,
	 int    padding,
	 float *eccentricities);

#define HANDLE_ERROR(err)						\
  do {									\
    if (err != cudaSuccess) {						\
      printf( __FILE__":%d : %s\n", __LINE__, cudaGetErrorString( err )); \
      exit( EXIT_FAILURE );						\
    }									\
  } while(0)

__host__ GSAPI int
gs_cuda_diameter(const GSMatrix *matrix)
{
  float ecc;
  int ind;
  int *degrees_device, *data_device;
  float *ecc_device, *ecc_host;
  dim3 block(16);
  dim3 grid(matrix->nodes / 16 + 1);
  cublasStatus status;

  status = cublasInit();
  
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! CUBLAS initialization error\n");
    exit(EXIT_FAILURE);
  }

  ecc = 0;

  HANDLE_ERROR(cudaMalloc((void**) &degrees_device, matrix->nodes * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**) &data_device,    matrix->size));
  HANDLE_ERROR(cudaMalloc((void**) &ecc_device,     matrix->nodes * sizeof(float)));

  ecc_host = (float*) malloc(matrix->nodes*sizeof(float));

  HANDLE_ERROR(cudaMemcpy(degrees_device, matrix->degrees, matrix->nodes * sizeof(int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(data_device,    matrix->cells,   matrix->size,                cudaMemcpyHostToDevice));

  diameter<<<grid, block>>>(matrix->nodes, degrees_device, data_device, matrix->davg, ecc_device);
  ind = cublasIsamax(matrix->nodes, ecc_device, 1);

  HANDLE_ERROR(cudaMemcpy(ecc_host, ecc_device, matrix->nodes * sizeof(float), cudaMemcpyDeviceToHost));

  for (ind = 0; ind < matrix->nodes; ind++)
    fprintf(stdout, "[%d] %f\n", ind, ecc_host[ind]);

  HANDLE_ERROR(cudaFree(degrees_device));
  HANDLE_ERROR(cudaFree(data_device));
  HANDLE_ERROR(cudaFree(ecc_device));

  status = cublasShutdown();
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! shutdown error (A)\n");
    exit(EXIT_FAILURE);
  }

  return ecc_host[ind];
}
