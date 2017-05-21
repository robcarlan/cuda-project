
////////////////////////////////////////////////////////////////////////
// GPU version of Monte Carlo algorithm using NVIDIA's CURAND library,
// With annotations.
////////////////////////////////////////////////////////////////////////

#include "cuda_headers.cuh"
#include "milstein_kernel.cuh"
#include "euler_kernel.cuh"

void regression(int, float *, float *, float &a, float &b);
void Check_CUDA_Error(const char *message);
int round_to_grid_size(int grid_size, int n);

float mlmc_gpu(
	int num_levels,
	int n_initial, int timesteps, float eps,
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

const int GRID_SIZE = 64;

// Note that the more complex version mlqmc06_l uses a Milstein method of discretisation which adds another term to Euler disc.
// Euler has strong order of convergence sqrt(delta(t)) compared to Milsteins delta(T)
template <int gsize>
__global__ void sum_reduce(double *d_v, double *d_v_sq) {
    __shared__  float temp[gsize];
    __shared__  float tempsq[gsize];
	
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

void pathcalc(int level, int samples, float *d_z, float *d_norm, float *d_unif,
	      double *d_v, double *d_v_sq,
	      bool gpu_reduce, bool use_milstein) {
	if (!use_milstein){
	    if (level == 0)
			pathcalc_half<<<samples / GRID_SIZE, GRID_SIZE>>>(d_z, d_v, d_v_sq);
	    if (level == 1)
			pathcalc_float<<<samples / GRID_SIZE, GRID_SIZE>>>(d_z, d_v, d_v_sq);
	    else
			pathcalc_double<<<samples / GRID_SIZE, GRID_SIZE>>>(d_z, d_v, d_v_sq);
	    Check_CUDA_Error("Euler Kernel Execution Failed!\n");		 
	} else {
	    if (level == 0)
	    	milstein_half<128><<<samples / GRID_SIZE, GRID_SIZE>>>(d_unif, d_norm, d_v, d_v_sq);
	    if (level == 1)
	    	milstein_float<128><<<samples / GRID_SIZE, GRID_SIZE>>>(d_unif, d_norm, d_v, d_v_sq);
	    else
	    	milstein_double<128><<<samples / GRID_SIZE, GRID_SIZE>>>(d_unif, d_norm, d_v, d_v_sq);
	    Check_CUDA_Error("Milstein Kernel Execution Failed!\n");
	}
	
	if (gpu_reduce) {
	     sum_reduce<64><<<samples / GRID_SIZE, GRID_SIZE>>>(d_v, d_v_sq);
	    
	    Check_CUDA_Error("GPU Reduce Kernel Failed!\n");
	  	 
	    //Optimisation - threadfence shuffles and stuff kernelise blocks
	}
}

////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////

float mlmc_gpu(
	int num_levels,
	int n_initial, int timesteps, float eps,
	float alpha_0, float beta_0, float gamma_0,
	int *out_samples_per_level, float *out_cost_per_level,
	int debug_level, bool use_timings, 
	bool gpu_reduce, bool milstein)
{
    int *Nl = out_samples_per_level;
    float *Cl = out_cost_per_level;
	
    if (gpu_reduce) printf("GPU reduce");

    int variant = 1;
    if (gpu_reduce) variant = 2;
    if (milstein) variant = 3;
    if (gpu_reduce && milstein) variant = 4;
    
  if (debug_level) {
      printf("CUDA multi-level monte carlo variant %d\n", variant);
#if __CUDA_ARCH__ >= 520
      printf("(CUDA half precision enabled...)\n");
#else
      printf("(CUDA half precision NOT enabled...)\n");
#endif
  }

  //This variant sets LMin and LMax set to be 2.
  int Lmin = 2;
  int Lmax = 2;

  //Number of timesteps.
  int h_N = 100;
  
  n_initial = round_to_grid_size(GRID_SIZE, n_initial);

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

  double ml[21], Vl[21], sums[7], suml[3][21];
  float  NlCl[21], x[21], y[21],
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
	  float  *d_norm, *d_unif;

	  //Allocate memory
	  h_v = (double *)malloc(sizeof(double) * num_paths);
	  h_v_sq = (double *)malloc(sizeof(double) * num_paths);
	  checkCudaErrors( cudaMalloc((void **)&d_v, sizeof(double)*num_paths) );
	  checkCudaErrors( cudaMalloc((void **)&d_v_sq, sizeof(double)*num_paths) );

	  //allocate memory depending on distribution needed
	  if (!milstein) {
	      checkCudaErrors( cudaMalloc((void **)&d_z, sizeof(float)*2*h_N*num_paths) );
	  } else {
	      checkCudaErrors( cudaMalloc((void **)&d_norm, sizeof(float)*4*h_N*num_paths) );
	      checkCudaErrors( cudaMalloc((void **)&d_unif, sizeof(float)*2*h_N*num_paths) );
	  }

	  if (debug_level)
	      printf("memory initialised level %d\n", l);
	  
	  if (use_timings)
	      cudaEventRecord(start);

	  //generate corresponding random number arrays
	  if (!milstein) {
	      checkCudaErrors( curandGenerateNormal(gen, d_z, 2*h_N*num_paths, 0.0f, 1.0f) );
	  } else {
	      checkCudaErrors( curandGenerateNormal(gen, d_norm, 4*h_N*num_paths, 0.0f, 1.0f) );
	      checkCudaErrors( curandGenerateUniform(gen, d_unif, 2*h_N*num_paths) );
	  }
	 
	  if (use_timings) {
	      cudaEventRecord(stop);
	      cudaEventSynchronize(stop);
	      cudaEventElapsedTime(&milli, start, stop);

	      printf("CURAND normal RNG  execution time (ms): %f,  samples/sec: %e \n",
		     milli, 2.0*h_N*num_paths/(0.001*milli));
	  }

	  if (debug_level)
	      printf("Runing kernel level %d grid_size %d num_paths %d\n", l, GRID_SIZE, num_paths);

	  cudaDeviceSynchronize();
	  pathcalc(l, num_paths, d_z, d_norm, d_unif, d_v, d_v_sq,
		   gpu_reduce, milstein);
	  cudaDeviceSynchronize();
		
	  if (debug_level)
	      printf("path calc level %d\n", l);

	  //Move results out of device memory, add to array.

	  checkCudaErrors( cudaMemcpy( (void *) h_v, (void *) d_v, sizeof(double)*num_paths,
				      cudaMemcpyDeviceToHost) );

	  checkCudaErrors( cudaMemcpy( (void *) h_v_sq, (void *) d_v_sq, sizeof(double)*num_paths,
				      cudaMemcpyDeviceToHost) );


	  //reduce step
	  if (!gpu_reduce) {
		for (int i = 0; i < num_paths; i++) {
		  // Number of timestep is 2^bit precision
		  sums[0] += 1 << l;
			
		  if (diag > 3)
		      printf("[%d,%d] val: %g val_sq: %g\n", l, i, h_v[i], h_v_sq[i]);
			
		  sums[1] += h_v[i];
		  sums[2] += h_v_sq[i];
		}
	  } else {
	      //Reduction has already been done on gpu, so just copy over.
	      //Sum over blocks.
	      for (int i = num_paths / GRID_SIZE - 2; i >= 0; i--) {
			  int thisIndex = i * GRID_SIZE;
			  int nextIndex = (i + 1) * GRID_SIZE;
			  printf("BLOCK %d %d %d\n", i, thisIndex, nextIndex);
			  printf("this %d next %d", thisIndex, nextIndex);
			  h_v[thisIndex] += h_v[nextIndex];
			  printf("HI");
			  h_v_sq[thisIndex] += h_v_sq[nextIndex];
			  printf("sq");
	      }
	      
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

	  if (!milstein) {
	      checkCudaErrors( cudaFree(d_z) );
	  } else {
	      checkCudaErrors( cudaFree(d_norm) );
	      checkCudaErrors( cudaFree(d_unif) );
	  }
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

      if (diag > 1)
		printf("ML %g \n" , suml[1][l]);
      ml[l] = abs(suml[1][l]/suml[0][l]);
      Vl[l] = max(suml[2][l]/suml[0][l] - ml[l]*ml[l], 0.0);

      if (diag > 0)
		printf("level %d: variance %.5g expectation %.5g\n", l, Vl[l], ml[l]);
      
      if (gamma_0 <= 0.0f) Cl[l] = NlCl[l] / suml[0][l];

      if (l>1) {
        ml[l] = max(ml[l],  0.5*ml[l-1]/pow(2.0,(double)alpha));
        Vl[l] = max(Vl[l],  0.5*Vl[l-1]/pow(2.0,(double)beta));
      }
	  
	  if (diag > 0)
		printf("level %d post check: variance %.5g expectation %.5g\n", l, Vl[l], ml[l]);

      sum += sqrt(Vl[l]*Cl[l]);
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
		printf(" level %d - %d (rounded to %d), ", l, dNl[l], round_to_grid_size(GRID_SIZE, dNl[l]));
      }
	  
	  dNl[l] = round_to_grid_size(GRID_SIZE, dNl[l]);
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
			dNl[l] = round_to_grid_size(GRID_SIZE, dNl[l]);
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
