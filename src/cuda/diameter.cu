
__global__ void diameter(int    nodes,
			 int   *degrees,
			 int   *data,
			 float *eccentricities)
{
  int index;
  index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < n) {
    int   looking, candidat, i, o, n, c;
    int   stack  [nodes];
    float closed [nodes];
    float d;

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
      
      neigh = cells + n * matrix->davg;
      
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

GSAPI int
gs_cuda_diameter(const matrix_t *matrix)
{
  float ecc;
  int ind;
  int *degrees_device, *data_device;
  float *ecc_device;
  dim3 block(16);
  dim3 grid(n / 16 + 1);

  cudaMalloc((void**) &degrees_device, matrix->nodes * sizeof(int));
  cudaMalloc((void**) &data_device,    matrix->size);
  cudaMalloc((void**) &ecc_device,     matrix->nodes * sizeof(float));

  cudaMemcpy(degrees_dev, matrix->degrees, matrix->nodes * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(data_dev,    matrix->data,    matrix->size,                cudaMemcpyHostToDevice);

  diameter<<<grid, block>>>(matrix->nodes, degrees_device, data_device, ecc_device);
  ind = cubasIsamax(matrix->nodes, ecc_device, 1);

  cudaMemcpy(&ecc, ecc_device + ind, sizeof(float), cudaMemcpyDeviceToHost);
}
