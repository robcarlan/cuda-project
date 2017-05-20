
////////////////////////////////////////////////////////////////////////
// GPU version of Monte Carlo algorithm using NVIDIA's CURAND library,
// With annotations.
////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include <curand.h>

//#if __CUDA_ARCH__ >= 530
#include <cuda_fp16.h>
//#pragma message("Compiling with CUDA half precision")
//#else
//#warning "CUDA half precision not supported!"
//#endif

#include <helper_cuda.h>

void regression(int, float *, float *, float &a, float &b);
void Check_CUDA_Error(const char *message);
int round_to_grid_size(int grid_size, int n);

float mlmc_gpu(
	int num_levels,
	int n_initial, float eps,
	float alpha_0, float beta_0, float gamma_0,
	int *out_samples_per_level, float *out_cost_per_level,
	int debug_level, bool use_timings, 
	bool gpu_reduce, bool milstein);
	
int round_to_grid_size(int grid_size, int n) {
	if (n % grid_size == 0) return n;
	else return (n / grid_size + 1)  * grid_size;
}

void Check_CUDA_Error(const char *message) {
	cudaError_t error = cudaGetLastError();
	if(error!=cudaSuccess) {
		fprintf(stderr,"ERROR: %s: %s\n", message, 
		cudaGetErrorString(error) );
		exit(-1);
	}
}

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

// Note that the more complex version mlqmc06_l uses a Milstein method of discretisation which adds another term to Euler disc.
// Euler has strong order of convergence sqrt(delta(t)) compared to Milsteins delta(T)

__global__ void sum_reduce(double *d_v, double *d_v_sq) {
	extern __shared__  float temp[];
	extern __shared__  float tempsq[];
	
    int tid = threadIdx.x + blockIdx.x*blockDim.x;

    // first, each thread loads data into shared memory
    temp[tid] = d_v[tid];
	tempsq[tid] = d_v_sq[tid];
	
	__syncthreads();

    // next, we perform binary tree reduction

    for (int d = blockDim.x>>1; d > 0; d >>= 1) {
      __syncthreads();  // ensure previous step completed 
      if (tid<d)  {
		  temp[tid] += temp[tid+d];
		  tempsq[tid] += tempsq[tid+d];
	  }
    }

    // finally, first thread puts result into global memory

    if (tid==0){
		d_v[0] = temp[0];
		d_v_sq[0] = tempsq[0];
	} 
}

//#if __CUDA_ARCH__ >= 530
__global__ void pathcalc_half(float *d_z, double *d_v, double *d_v_sq)
{

  __half one = __float2half(1.0f);
  __half point1 = __float2half(0.1f);
  __half negpoint1 = __float2half(-0.1f);

  __half s1, s2, y1, y2;
  float payoff = 0.0f;
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
      payoff = __half2float(hexp(__hmul(__float2half((float)-r_dbl),
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

    y2   = rho_dbl *y1 + alpha_dbl * d_z[ind];
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

template <bool gpu_reduce>
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

void pathcalc(int level, int gsize, int samples, float *d_z, double *d_v, double *d_v_sq,
		bool gpu_reduce, bool use_milstein) {
	if (!use_milstein){
		if (level == 0)
			pathcalc_half<<<samples / gsize, gsize>>>(d_z, d_v, d_v_sq);
		if (level == 1)
			pathcalc_float<<<samples / gsize, gsize>>>(d_z, d_v, d_v_sq);
		if (level == 2)
			pathcalc_double<false><<<samples / gsize, gsize>>>(d_z, d_v, d_v_sq);
		 Check_CUDA_Error("Euler Kernel Execution Failed!\n");
		 
	} else {
		pathcalc_float<<<samples / gsize, gsize>>>(d_z, d_v, d_v_sq);
		
		 Check_CUDA_Error("Milstein Kernel Execution Failed!\n");
	}
	
	if (gpu_reduce) {
		sum_reduce<<<samples / gsize, gsize, 2 * sizeof(double)>>>(d_v, d_v_sq);
		
		 Check_CUDA_Error("GPU Reduce Kernel Failed!\n");
		 
		 //Sum over blocks.
		 
		 //Optimisation - threadfence shuffles and stuff kernelise blocks
	}
}

////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////

float mlmc_gpu(
	int num_levels,
	int n_initial, float eps,
	float alpha_0, float beta_0, float gamma_0,
	int *out_samples_per_level, float *out_cost_per_level,
	int debug_level, bool use_timings, 
	bool gpu_reduce, bool milstein)
{
    int *Nl = out_samples_per_level;
    float *Cl = out_cost_per_level;
	
	if (gpu_reduce) printf("GPU reduce");

  if (debug_level) {
      printf("CUDA multi-level monte carlo variant 1\n");
#if __CUDA_ARCH__ >= 520
      printf("(CUDA half precision enabled...)\n");
#else
      printf("(CUDA half precision NOT enabled...)\n");
#endif
  }

  //This variant sets LMin and LMax set to be 2.
  int Lmin = 2;
  int Lmax = 2;
  int grid_size = 64;

  //Number of timesteps.
  int h_N = 100;
  
  n_initial = round_to_grid_size(grid_size, n_initial);

  double   h_T, h_r, h_sigma, h_rho, h_alpha, h_dt, h_con1, h_con2;

  h_T     = 1.0;
  h_r     = 0.05;
  h_sigma = 0.1;
  h_rho   = 0.5;
  h_alpha = sqrt(1.0-h_rho*h_rho);
  h_dt    = 1.0/ 128;
  h_con1  = 1.0 + h_r*h_dt;
  h_con2  = sqrt(h_dt)*h_sigma;

  checkCudaErrors( cudaMemcpyToSymbol(N,                &h_N,                           sizeof(h_N)) );
  checkCudaErrors( cudaMemcpyToSymbol(T_dbl,    	&h_T,    			sizeof(h_T)) );
  checkCudaErrors( cudaMemcpyToSymbol(r_dbl,    	&h_r,    			sizeof(h_r)) );
  checkCudaErrors( cudaMemcpyToSymbol(sigma_dbl,	&h_sigma,			sizeof(h_sigma)) );
  checkCudaErrors( cudaMemcpyToSymbol(rho_dbl,  	&h_rho,  			sizeof(h_rho)) );
  checkCudaErrors( cudaMemcpyToSymbol(alpha_dbl ,	&h_alpha,			sizeof(h_alpha)) );
  checkCudaErrors( cudaMemcpyToSymbol(dt_dbl,   	&h_dt,   			sizeof(h_dt)) );
  checkCudaErrors( cudaMemcpyToSymbol(con1_dbl, 	&h_con1, 			sizeof(h_con1)) );
  checkCudaErrors( cudaMemcpyToSymbol(con2_dbl, 	&h_con2, 			sizeof(h_con2)) );

  double sums[7], suml[3][21];
  float  ml[21], Vl[21], NlCl[21], x[21], y[21],
         alpha, beta, gamma, sum, theta;
  int    dNl[21], L, converged;

  int    diag = debug_level;  // diagnostics, set to 0 for none
  
  //
  // check input parameters
  //

  if (num_levels < 1) {
    fprintf(stderr,"error: needs num_levels >= 1 \n");
    exit(1);
  }

  //
  // initialisation
  //

  // initialise CUDA timing

  float milli;
  cudaEvent_t start, stop;

  if (use_timings) {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  curandGenerator_t gen;
  checkCudaErrors( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
  checkCudaErrors( curandSetPseudoRandomGeneratorSeed(gen, 1234ULL) );
	
  alpha = fmax(0.0f,alpha_0);
  beta  = fmax(0.0f,beta_0);
  gamma = fmax(0.0f,gamma_0);
  theta = 0.25f;             // MSE split between bias^2 and variance

  L = Lmin;
  converged = 0;

  for(int l=0; l<=Lmax; l++) {
    Nl[l]   = 0;
    Cl[l]   = powf(2.0f,(float)l*gamma);
    NlCl[l] = 0.0f;

    for(int n=0; n<3; n++) suml[n][l] = 0.0;
  }

  for(int l=0; l<=Lmin; l++) dNl[l] = n_initial;

  if (diag > 1)
      printf("Initialised - entering main loop.\n");

  //
  // main loop
  //

  while (!converged) {

    //
    // update sample sums
    //

    for (int l=0; l<=L; l++) {
      if (diag) printf(" %d ",dNl[l]);

      if (dNl[l]>0) {
		  
		sums[0] = sums[1] = sums[2] = 0.0f;

    	int num_paths = dNl[l];

    	double *h_v, *d_v;
    	double *h_v_sq, *d_v_sq;
    	float  *d_z;

    	//Allocate memory
		
		//TODO :: should be the same for each level.
    	h_v = (double *)malloc(sizeof(double) * num_paths);
    	h_v_sq = (double *)malloc(sizeof(double) * num_paths);
    	checkCudaErrors( cudaMalloc((void **)&d_v, sizeof(double)*num_paths) );
    	checkCudaErrors( cudaMalloc((void **)&d_v_sq, sizeof(double)*num_paths) );
    	checkCudaErrors( cudaMalloc((void **)&d_z, sizeof(float)*2*h_N*num_paths) );

		if (debug_level)
			printf("memory initialised level %d\n", l);

		//Generate 2 * dNl[l] random samples at desired precision based on l.
		if (use_timings)
			cudaEventRecord(start);

		checkCudaErrors( curandGenerateNormal(gen, d_z, 2*h_N*num_paths, 0.0f, 1.0f) );
	 
		if (use_timings) {
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&milli, start, stop);

			printf("CURAND normal RNG  execution time (ms): %f,  samples/sec: %e \n",
			   milli, 2.0*h_N*num_paths/(0.001*milli));
		}

		if (debug_level)
			printf("Runing kernel level %d grid_size %d num_paths %d\n", l, grid_size, num_paths);

		cudaDeviceSynchronize();
		pathcalc(l, grid_size, num_paths, d_z, d_v, d_v_sq,
			gpu_reduce, milstein);
		cudaDeviceSynchronize();
		
		if (debug_level)
			printf("path calc level %d\n", l);

		//Move results out of device memory, add to array.

		checkCudaErrors( cudaMemcpy(h_v, d_v, sizeof(double)*num_paths,
						 cudaMemcpyDeviceToHost) );

		checkCudaErrors( cudaMemcpy(h_v_sq, d_v_sq, sizeof(double)*num_paths,
						 cudaMemcpyDeviceToHost) );

		if (debug_level)
			printf("reduce step\n");

		//reduce step
		if (!gpu_reduce) {
			for (int i = 0; i < num_paths; i++) {
				// Number of timestep is 2^bit precision
				sums[0] += 1 << l;
			
				if (diag > 3)
					printf("[%d,%d] val: %f val_sq: %f\n", l, i, h_v[i], h_v_sq[i]);
			
				sums[1] += h_v[i];
				sums[2] += h_v_sq[i];
			}
		} else {
			//Reduction has already been done on gpu, so just copy over.
			sums[0] += (1 << l) * num_paths;
			sums[1] += h_v[0];
			sums[2] += h_v_sq[0];
		}

		if (debug_level)
			printf("reduce completed %d - %g %g \n", l, sums[1], sums[2]);

		suml[0][l] += (double) num_paths;
		suml[1][l] += sums[1];
		suml[2][l] += sums[2];
		NlCl[l]    += sums[0];  // sum total cost

		//Free used memory
		free(h_v);
		free(h_v_sq);
		checkCudaErrors( cudaFree(d_v) );
		checkCudaErrors( cudaFree(d_v_sq) );
		checkCudaErrors( cudaFree(d_z) );
      }
    }
    if (diag) printf(" \n");

    //
    // compute absolute average, variance and cost,
    // correct for possible under-sampling,
    // and set optimal number of new samples
    //

    sum = 0.0f;

    for (int l=0; l<=L; l++) {
	  printf("ML %g \n" , suml[1][l]);
      ml[l] = fabs(suml[1][l]/suml[0][l]);
      Vl[l] = fmaxf(suml[2][l]/suml[0][l] - ml[l]*ml[l], 0.0f);

      if (diag > 2)
		printf("level %d: variance %.5f expectation %.5f\n", l, Vl[l], ml[l]);
      
      if (gamma_0 <= 0.0f) Cl[l] = NlCl[l] / suml[0][l];

      if (l>1) {
        ml[l] = fmaxf(ml[l],  0.5f*ml[l-1]/powf(2.0f,alpha));
        Vl[l] = fmaxf(Vl[l],  0.5f*Vl[l-1]/powf(2.0f,beta));
      }

      sum += sqrtf(Vl[l]*Cl[l]);
    }

    if (diag > 1) {
		printf("Next level samples: ");
    }
    
    //Now update the number of samples for each level.
    for (int l=0; l<=L; l++) {
      dNl[l] = ceilf( fmaxf( 0.0f,
                       sqrtf(Vl[l]/Cl[l])*sum/((1.0f-theta)*eps*eps)
                     - suml[0][l] ) );
      if (diag > 1) {
		printf(" level %d - %d (rounded to %d), ", l, dNl[l], round_to_grid_size(grid_size, dNl[l]));
      }
	  
	  dNl[l] = round_to_grid_size(grid_size, dNl[l]);
    }
   
    if (diag > 1) {
	printf("\n");
    }
 

    //
    // use linear regression to estimate alpha, beta, gamma if not given
    //

    if (alpha_0 <= 0.0f) {
      for (int l=1; l<=L; l++) {
        x[l-1] = l;
        y[l-1] = - log2f(ml[l]);
      }
      regression(L,x,y,alpha,sum);
      if (diag) printf(" alpha = %f \n",alpha);
    }

    if (beta_0 <= 0.0f) {
      for (int l=1; l<=L; l++) {
        x[l-1] = l;
        y[l-1] = - log2f(Vl[l]);
      }
      regression(L,x,y,beta,sum);
      if (diag) printf(" beta = %f \n",beta);
    }

     if (gamma_0 <= 0.0f) {
      for (int l=1; l<=L; l++) {
        x[l-1] = l;
        y[l-1] = log2f(Cl[l]);
      }
      regression(L,x,y,gamma,sum);
      if (diag) printf(" gamma = %f \n",gamma);
    }

    //
    // if (almost) converged, estimate remaining error and decide
    // whether a new level is required
    //

    sum = 0.0;
      for (int l=0; l<=L; l++)
        sum += fmaxf(0.0f, (float)dNl[l]-0.01f*suml[0][l]);

    if (sum==0) {
      if (diag) printf(" achieved variance target \n");

      converged = 1;
      float rem = ml[L] / (powf(2.0f,gamma)-1.0f);

      if (rem > sqrtf(theta)*eps) {
        if (L==Lmax)
          printf("*** failed to achieve weak convergence *** \n");
        else {
          converged = 0;
          L++;
          Vl[L] = Vl[L-1]/powf(2.0f,beta);
          Cl[L] = Cl[L-1]*powf(2.0f,gamma);

          if (diag) printf(" L = %d \n",L);

          sum = 0.0f;
          for (int l=0; l<=L; l++) sum += sqrtf(Vl[l]*Cl[l]);
          for (int l=0; l<=L; l++) {
            dNl[l] = ceilf( fmaxf( 0.0f,
                            sqrtf(Vl[l]/Cl[l])*sum/((1.0f-theta)*eps*eps)
                          - suml[0][l] ) );
			dNl[l] = round_to_grid_size(grid_size, dNl[l]);
		  }
        }
      }
    }
  }

  //
  // finally, evaluate multilevel estimator and set outputs
  //

  float P = 0.0f;
  for (int l=0; l<=L; l++) {
    P    += suml[1][l]/suml[0][l];
    Nl[l] = suml[0][l];
    Cl[l] = NlCl[l] / Cl[l];
  }

  return P;
}



//
// linear regression routine
//

void regression(int N, float *x, float *y, float &a, float &b){

  float sum0=0.0f, sum1=0.0f, sum2=0.0f, sumy0=0.0f, sumy1=0.0f;

  for (int i=0; i<N; i++) {
    sum0  += 1.0f;
    sum1  += x[i];
    sum2  += x[i]*x[i];

    sumy0 += y[i];
    sumy1 += y[i]*x[i];
  }

  a = (sum0*sumy1 - sum1*sumy0) / (sum0*sum2 - sum1*sum1);
  b = (sum2*sumy0 - sum1*sumy1) / (sum0*sum2 - sum1*sum1);
}

/*

int main(int argc, const char **argv){
    
  int     NPATH=960000, h_N=100;
  float   h_T, h_r, h_sigma, h_rho, h_alpha, h_dt, h_con1, h_con2;
  float  *h_v, *d_v, *d_z;
  double  sum1, sum2;

  // initialise card

  findCudaDevice(argc, argv);

  // initialise CUDA timing

  float milli;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

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

  cudaEventRecord(start);

  curandGenerator_t gen;
  checkCudaErrors( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
  checkCudaErrors( curandSetPseudoRandomGeneratorSeed(gen, 1234ULL) );
  checkCudaErrors( curandGenerateNormal(gen, d_z, 2*h_N*NPATH, 0.0f, 1.0f) );
 
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);

  printf("CURAND normal RNG  execution time (ms): %f,  samples/sec: %e \n",
          milli, 2.0*h_N*NPATH/(0.001*milli));

  // execute kernel and time it

  cudaEventRecord(start);

  //pathcalc<<<NPATH/64, 64>>>(d_z, d_v);
  getLastCudaError("pathcalc execution failed\n");

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);

  printf("Monte Carlo kernel execution time (ms): %f \n",milli);

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

}

*/
