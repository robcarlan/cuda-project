

////////////////////////////////////////////////////////////////////////
// GPU version of Monte Carlo algorithm using NVIDIA's CURAND library
////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include <curand.h>
#include <cuda_fp16.h>

#include "fp16_conversion"
#include "helper_cuda.h"

////////////////////////////////////////////////////////////////////////
// CUDA global constants
////////////////////////////////////////////////////////////////////////

__constant__ int   N;
__constant__ float T, r, sigma, rho, alpha, dt, con1, con2;

////////////////////////////////////////////////////////////////////////
// kernel routine basic. As provided in practical 2.
////////////////////////////////////////////////////////////////////////


__global__ void pathcalc(float *d_z, float *d_v)
{
  float s1, s2, y1, y2, payoff;
  int   ind;

  // move array pointers to correct position

  // version 1
  ind = threadIdx.x + 2*N*blockIdx.x*blockDim.x;


  // path calculation

  s1 = 1.0f;
  s2 = 1.0f;

  for (int n=0; n<N; n++) {
    y1   = d_z[ind];
    ind += blockDim.x;      // shift pointer to next element

    y2   = rho*y1 + alpha*d_z[ind];
    ind += blockDim.x;      // shift pointer to next element

    s1 = s1*(con1 + con2*y1);
    s2 = s2*(con1 + con2*y2);
  }

  // put payoff value into device array
  payoff = 0.0f;
  if ( fabs(s1-1.0f)<0.1f && fabs(s2-1.0f)<0.1f ) payoff = exp(-r*T);

  d_v[threadIdx.x + blockIdx.x*blockDim.x] = payoff;
}

////////////////////////////////////////////////////////////////////////
// kernel routine level path calc. 
////////////////////////////////////////////////////////////////////////

#define pathcalc_low pathcalc_level<1>
#define pathcalc_mid pathcalc_level<2>
#define pathcalc_high pathcalc_level<3>

template <int level>
__global__ void pathcalc_level(float *d_z, float *d_v)
{
  float s1, s2, y1, y2, payoff;
  int   ind;

  // move array pointers to correct position

  // version 1
  ind = threadIdx.x + 2*N*blockIdx.x*blockDim.x;


  // path calculation

  s1 = 1.0f;
  s2 = 1.0f;

  for (int n=0; n<N; n++) {
    y1   = d_z[ind];
    ind += blockDim.x;      // shift pointer to next element

    y2   = rho*y1 + alpha*d_z[ind];
    ind += blockDim.x;      // shift pointer to next element

    s1 = s1*(con1 + con2*y1);
    s2 = s2*(con1 + con2*y2);
  }

  // put payoff value into device array
  payoff = 0.0f;
  if ( fabs(s1-1.0f)<0.1f && fabs(s2-1.0f)<0.1f ) payoff = exp(-r*T);

  d_v[threadIdx.x + blockIdx.x*blockDim.x] = payoff;
}

////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////

/*
   P = mlmc(Lmin,Lmax,N0,eps, mlmc_l, alpha,beta,gamma, Nl)
 
   multilevel Monte Carlo control routine

   num_levels  = levels of refinement       >= 2
   n_initial    = initial number of samples         > 0
   epsilon   = desired accuracy (rms error)      > 0 
 
   alpha -> weak error is  O(2^{-alpha*l})
   beta  -> variance is    O(2^{-beta*l})
   gamma -> sample cost is O(2^{gamma*l})

   if alpha, beta, gamma are not positive then they will be estimated

   mlmc_l(l,N,sums)   low-level function
        l       = level
        N       = number of paths
        sums[0] = sum(cost)
        sums[1] = sum(Y)
        sums[2] = sum(Y.^2)
        where Y are iid samples with expected value:
        E[P_0]           on level 0
        E[P_l - P_{l-1}] on level l>0

   P     = value
   Nl    = number of samples at each level
   NlCl  = total cost of samples at each level

*/

int mlmc_gpu(
	int num_levels,
	int n_initial, float epsilon, 
	float alpha_0, float beta_0, float gamma_0, 
	int &out_samples_per_level, float &out_cost_per_level,
	bool use_debug, bool use_timings)
			{
    
  int     NPATH=960000, h_N=100;
  float   h_T, h_r, h_sigma, h_rho, h_alpha, h_dt, h_con1, h_con2;
  float  *h_v, *d_v, *d_z;
  double  sum1, sum2;

  // initialise card

  const char * arg = "hi";
  const char ** argv = &arg;
  findCudaDevice(0, argv);

  // initialise CUDA timing

  float milli;
  cudaEvent_t start, stop;

  if (use_timings) {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  // allocate memory on host and device

  h_v = (float *)malloc(sizeof(float)*NPATH);

  checkCudaErrors( cudaMalloc((void **)&d_v, sizeof(float)*NPATH) );
  checkCudaErrors( cudaMalloc((void **)&d_z, sizeof(float)*2*h_N*NPATH) );

  // define constants and transfer to GPU

  h_T     = 1.0f;
  h_r     = 0.05f;
  h_sigma = 0.1f;
  h_rho   = 0.5f;
  h_alpha = sqrt(1.0f-h_rho*h_rho);
  h_dt    = 1.0f/h_N;
  h_con1  = 1.0f + h_r*h_dt;
  h_con2  = sqrt(h_dt)*h_sigma;

  checkCudaErrors( cudaMemcpyToSymbol(N,    &h_N,    sizeof(h_N)) );
  checkCudaErrors( cudaMemcpyToSymbol(T,    &h_T,    sizeof(h_T)) );
  checkCudaErrors( cudaMemcpyToSymbol(r,    &h_r,    sizeof(h_r)) );
  checkCudaErrors( cudaMemcpyToSymbol(sigma,&h_sigma,sizeof(h_sigma)) );
  checkCudaErrors( cudaMemcpyToSymbol(rho,  &h_rho,  sizeof(h_rho)) );
  checkCudaErrors( cudaMemcpyToSymbol(alpha,&h_alpha,sizeof(h_alpha)) );
  checkCudaErrors( cudaMemcpyToSymbol(dt,   &h_dt,   sizeof(h_dt)) );
  checkCudaErrors( cudaMemcpyToSymbol(con1, &h_con1, sizeof(h_con1)) );
  checkCudaErrors( cudaMemcpyToSymbol(con2, &h_con2, sizeof(h_con2)) );

  // random number generation

  // - all random numbers are generated before hand and stored in
  // the array d_z, which will be used by the kernel.
  // Kernel speed then depends on the method of which indexing method
  // to access by.

  if (use_timings)
    cudaEventRecord(start);

  curandGenerator_t gen;
  checkCudaErrors( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
  checkCudaErrors( curandSetPseudoRandomGeneratorSeed(gen, 1234ULL) );
  checkCudaErrors( curandGenerateNormal(gen, d_z, 2*h_N*NPATH, 0.0f, 1.0f) );
 
  if (use_timings) {
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milli, start, stop);

    printf("CURAND normal RNG  execution time (ms): %f,  samples/sec: %e \n",
            milli, 2.0*h_N*NPATH/(0.001*milli));
  }

  // execute kernel and time it

  cudaEventRecord(start);

  pathcalc_low<<<NPATH/64, 64>>>(d_z, d_v);
  getLastCudaError("pathcalc execution failed\n");

  if (use_timings) {
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milli, start, stop);

    printf("Monte Carlo kernel execution time (ms): %f \n",milli);
  }

  // copy back results
  
  checkCudaErrors( cudaMemcpy(h_v, d_v, sizeof(float)*NPATH,
                   cudaMemcpyDeviceToHost) );

  // compute average

  sum1 = 0.0;
  sum2 = 0.0;
  for (int i=0; i<NPATH; i++) {
    sum1 += h_v[i];
    sum2 += h_v[i]*h_v[i];
  }

  printf("\nAverage value and standard deviation of error  = %13.8f %13.8f\n\n",
	 sum1/NPATH, sqrt((sum2/NPATH - (sum1/NPATH)*(sum1/NPATH))/NPATH) );

  // Tidy up library

  checkCudaErrors( curandDestroyGenerator(gen) );

  // Release memory and exit cleanly

  free(h_v);
  checkCudaErrors( cudaFree(d_v) );
  checkCudaErrors( cudaFree(d_z) );

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();

  return 0;
}
