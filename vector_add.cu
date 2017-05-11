//
// include files
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "helper_cuda.h"


//
// kernel routine
// 

__global__ void my_first_kernel(float *x)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  x[tid] = (float) threadIdx.x;
}

__global__ void add_kernel(float *res, float *v1, float *v2)
{
	int tid = threadIdx.x + blockDim.x*blockIdx.x;

	res[tid] = v1[tid] + v2[tid];

}


//
// main code
//

int main(int argc, const char **argv)
{
  float *vec1_h, *vec2_h;
  float *vec1_d, *vec2_d;
  float *res_d, *res_h;
  int   nblocks, nthreads, nsize, n; 

  // initialise card

  findCudaDevice(argc, argv);

  // set number of blocks, and threads per block

  nblocks  = 2;
  nthreads = 4;
  nsize    = nblocks*nthreads ;

  // allocate memory for array

  vec1_h = (float *)malloc(nsize*sizeof(float));
  vec2_h = (float *)malloc(nsize*sizeof(float));
  res_h = (float *)malloc(nsize*sizeof(float));
  checkCudaErrors(cudaMalloc((void **)&vec1_d, nsize*sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&vec2_d, nsize*sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&res_d, nsize*sizeof(float)));
  
  //Initiliase vectors

  for (int i = 0; i < nsize; i++) {
	  vec1_h[i] = (float)i;
	  vec2_h[i] = (float)i;
	  printf("vec1(%d): %f, vec2(%d): %f \n",
			  i, vec1_h[i],
			  i, vec2_h[i]);
  }

  //copy host vectors to device vectors

  checkCudaErrors( cudaMemcpy(vec1_d,vec1_h,nsize*sizeof(float),
                 cudaMemcpyHostToDevice) );

  checkCudaErrors( cudaMemcpy(vec2_d,vec2_h,nsize*sizeof(float),
                 cudaMemcpyHostToDevice) );

  add_kernel<<<nblocks,nthreads>>>(res_d, vec1_d, vec2_d);
  getLastCudaError("add_kernel execution failed\n");

  // copy back results and print them out

  checkCudaErrors( cudaMemcpy(res_h,res_d,nsize*sizeof(float),
                 cudaMemcpyDeviceToHost) );

  for (n=0; n<nsize; n++) printf(" sum(%d) : %f \n",n,res_h[n]);

  // free memory 

  checkCudaErrors(cudaFree(res_d));
  checkCudaErrors(cudaFree(vec1_d));
  checkCudaErrors(cudaFree(vec2_d));
  free(vec1_h);
  free(vec2_h);
  free(res_h);

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();

  return 0;
}
