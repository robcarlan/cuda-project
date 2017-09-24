#pragma once
#include "cuda_headers.cuh"

////////////////////////////////////////////////////////////////////////
// CUDA global constants
////////////////////////////////////////////////////////////////////////

__constant__ int   N;
// Store double constants and replace with kernel casts
__constant__ double T_dbl, r_dbl, sigma_dbl, rho_dbl, alpha_dbl, dt_dbl, con1_dbl, con2_dbl;

////////////////////////////////////////////////////////////////////////
// kernel routine
////////////////////////////////////////////////////////////////////////

// Attempts to solve 2D geometric brownian motion SDE:
//   dS_1 = r * S_1 * dt + rho * S_1 * dW_1
//   dS_2 = r * S_2 * dt + rho * S_2 * dW_2
// dW_1 and dW_2 are increments in two correlated brownian motions.

// This is approximated using Euler-Maruyama discretisation:
//   S_1,N+1 = S_1,N * (1 + r*delta(T) + rho * sqrt(delta(T)) * Y_1, N
//   S_2,N+1 = S_2,N * (1 + r*delta(T) + rho * sqrt(delta(T)) * Y_2, N

// delta(T) is the timestep. Y1_N and Y2_N are Normal r.v.
// Independent with other timesteps, but have correlation p which can be simulated by defining them as:
//   Y_1,N = Z_1,N
//   Y_2,N = rho * Z_1,N + sqrt(1-rho^2) * Z_2,N

//#if __CUDA_ARCH__ >= 530
__global__ void pathcalc_half(float *d_z, double *d_v, double *d_v_sq)
{

  __half one = __float2half(1.0f);
  __half point1 = __float2half(0.1f);
  __half negpoint1 = __float2half(-0.1f);

  __half s1, s2, y1, y2;
  double payoff = 0.0f;
  int   ind;

  // move array pointers to correct position

  ind = threadIdx.x + 2*N*blockIdx.x*blockDim.x;

  // path calculation

  s1 = one;
  s2 = s1;

  for (int n=0; n<N; n++) {
    y1   = __float2half(d_z[ind]);
    ind += blockDim.x;      // shift pointer to next element

    y2   = __hfma(__float2half((float)rho_dbl), y1,
		    __hmul(__float2half((float)alpha_dbl), __float2half(d_z[ind])));
    ind += blockDim.x;      // shift pointer to next element

    s1 = __hmul(s1, (__hfma(__float2half((float)con2_dbl), y1, __float2half((float)con1_dbl))));
    s2 = __hmul(s2, (__hfma(__float2half((float)con2_dbl), y2, __float2half((float)con1_dbl))));
  
  }

  // put payoff value into device array

  payoff = 0.0f;
  __half s1diff = __hsub(s1, one);
  __half s2diff = __hsub(s2, one);

  if ( 	__hgt(s1diff, negpoint1) && __hlt(s1diff, point1) &&
		    __hgt(s2diff, negpoint1) && __hlt(s2diff, point1) )
      payoff = (double)__half2float(hexp(__hmul(__float2half((float)(-r_dbl)),
					__float2half((float)T_dbl))) );
  
  d_v[threadIdx.x + blockIdx.x*blockDim.x] = payoff;
  d_v_sq[threadIdx.x + blockIdx.x*blockDim.x] = payoff * payoff;
}
//#else 
//#define pathcalc_half pathcalc_float
//#endif

//#if __CUDA_ARCH__ >= 530
__global__ void pathcalc_float(float *d_z, double *d_v, double *d_v_sq)
{
  float s1, s2, y1, y2, payoff, payoffh;
  __half s1h, s2h, y1h, y2h;
  int   ind;

  // move array pointers to correct position
  ind = threadIdx.x + 2*N*blockIdx.x*blockDim.x;
  
  __half one = __float2half(1.0f);
  __half point1 = __float2half(0.1f);
  __half negpoint1 = __float2half(-0.1f);

  // path calculation
  s1 = 1.0f;
  s2 = 1.0f;
  s1h = one;
  s2h = s1h;
  
  for (int n=0; n<N; n++) {
    y1   = d_z[ind];
    ind += blockDim.x;      // shift pointer to next element

    y2   = rho_dbl * y1 + alpha_dbl * d_z[ind];
    ind += blockDim.x;      // shift pointer to next element

    s1 = s1*(con1_dbl + con2_dbl *y1);
    s2 = s2*(con1_dbl + con2_dbl *y2);
	
    y1h   = __float2half(d_z[ind]);
    y2h   = __hfma(__float2half((float)rho_dbl), y1h,
		        __hmul(__float2half((float)alpha_dbl), __float2half(d_z[ind])));
    s1h = __hmul(s1h, (__hfma(__float2half((float)con2_dbl), y1h, __float2half((float)con1_dbl))));
    s2h = __hmul(s2h, (__hfma(__float2half((float)con2_dbl), y2h, __float2half((float)con1_dbl))));
  }
 
  // put payoff value into device array

  payoff = 0.0f;
  payoffh = 0.0f;
  if ( fabs(s1-1.0f)<0.1f && fabs(s2-1.0f)<0.1f ) payoff = exp(-r_dbl * T_dbl);
  
  __half s1diff = __hsub(s1h, one);
  __half s2diff = __hsub(s2h, one);
  
  if (__hgt(s1diff, negpoint1) && __hlt(s1diff, point1) &&
      __hgt(s2diff, negpoint1) && __hlt(s2diff, point1) )
      payoffh = __half2float(
	    hexp(__hmul(
		    __float2half((float)(-r_dbl)),
		    __float2half((float)T_dbl))));		

  d_v[threadIdx.x + blockIdx.x*blockDim.x] = payoff - payoffh;
  d_v_sq[threadIdx.x + blockIdx.x*blockDim.x] = (payoff - payoffh) * (payoff - payoffh);

}
//#else 
//__global__ void pathcalc_float(float *d_z, double *d_v, double *d_v_sq)
//{
	//As estimators would be the same! Not great at all.
//  d_v[threadIdx.x + blockIdx.x*blockDim.x] = 0.0;
//  d_v_sq[threadIdx.x + blockIdx.x*blockDim.x] = 0.0;
//}	
//#endif

__global__ void pathcalc_double(float *d_z, double *d_v, double *d_v_sq)
{
  double s1, s2, y1, y2, payoff;
  float s1f, s2f, y1f, y2f, payofff;
  int   ind;

  // move array pointers to correct position
  ind = threadIdx.x + 2*N*blockIdx.x*blockDim.x;

  // path calculation
  s1 = 1.0;
  s2 = 1.0;
  s1f = 1.0;
  s2f = 1.0;

  for (int n=0; n<N; n++) {
    y1   = d_z[ind];
	  y1f   = d_z[ind];
    ind += blockDim.x;      // shift pointer to next element

    y2   = rho_dbl *y1 + alpha_dbl *d_z[ind];
	  y2f   = (float)rho_dbl * y1f + alpha_dbl * d_z[ind];
    ind += blockDim.x;      // shift pointer to next element

    s1 = s1*(con1_dbl + con2_dbl *y1);
    s2 = s2*(con1_dbl + con2_dbl *y2);
	  s1f = s1f*(con1_dbl + con2_dbl *y1f);
    s2f = s2f*(con1_dbl + con2_dbl *y2f);
  }

  // put payoff value into device array

  payoff = payofff = 0.0f;
  if ( abs(s1-1.0)<0.1 && abs(s2-1.0)<0.1 ) payoff = exp(-r_dbl * T_dbl);
  if ( abs(s1f-1.0)<0.1 && abs(s2f-1.0)<0.1 ) payofff = exp(-r_dbl * T_dbl);

  d_v[threadIdx.x + blockIdx.x*blockDim.x] = payoff - payofff;
  d_v_sq[threadIdx.x + blockIdx.x*blockDim.x] = (payoff - payofff) * (payoff - payofff);  
  
  //printf("Dif: %g \n",d_v[threadIdx.x + blockIdx.x*blockDim.x] );
}
