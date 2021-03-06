/*
   P = mlmc(Lmin,Lmax,N0,eps, mlmc_l, alpha,beta,gamma, Nl)
 
   multilevel Monte Carlo control routine

   Lmin  = minimum level of refinement       >= 2
   Lmax  = maximum level of refinement       >= Lmin
   N0    = initial number of samples         > 0
   eps   = desired accuracy (rms error)      > 0 
 
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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void mcqmc06_l(int l, int N, double *sums); 

float mlmc(int Lmin, int Lmax, int N0, float eps,
      float alpha_0,float beta_0,float gamma_0, int *Nl, float *Cl,
      bool use_debug
    );

extern void regression(int, float *, float *, float &a, float &b);

float mlmc_cpu(int num_levels,
      int n_initial, float epsilon,
      float alpha_0, float beta_0, float gamma_0,
      int *out_samples_per_level, float *out_cost_per_level,
      bool use_debug, bool use_timings) {
    return mlmc(2, num_levels, n_initial, epsilon,
		    alpha_0, beta_0, gamma_0,
		    out_samples_per_level, out_cost_per_level, 
		    use_debug);
}


float mlmc(int Lmin, int Lmax, int N0, float eps,
      float alpha_0,float beta_0,float gamma_0, int *Nl, float *Cl, 
      bool use_debug) {

  double sums[7], suml[3][21];
  float  ml[21], Vl[21], NlCl[21], x[21], y[21],
         alpha, beta, gamma, sum, theta;
  int    dNl[21], L, converged;

  int    diag = use_debug;  // diagnostics, set to 0 for none 

  //
  // check input parameters
  //

  if (Lmin<2) {
      fprintf(stderr,"error: needs Lmin >= 2 \n");
      exit(1);
  }
  if (Lmax<Lmin) {
      fprintf(stderr,"error: needs Lmax >= Lmin \n");
      exit(1);
  }

  if (N0<=0 || eps<=0.0f) {
      fprintf(stderr,"error: needs N>0, eps>0 \n");
      exit(1);
  }

  //
  // initialisation
  //

  alpha = fmax(0.0f,alpha_0);
  beta  = fmax(0.0f,beta_0);
  gamma = fmax(0.0f,gamma_0);
  theta = 0.25f;             // MSE split between bias^2 and variance

  L = Lmin;
  converged = 0;

  for(int l=0; l<=Lmax; l++) {
    Nl[l]   = 0;

    //Cost is exponential by 2 ^^ level * gamma. Is this the case for precision?
    Cl[l]   = powf(2.0f,(float)l*gamma);
    NlCl[l] = 0.0f;

    for(int n=0; n<3; n++) suml[n][l] = 0.0;
  }

  //Number of samples is initially N0.
  for(int l=0; l<=Lmin; l++) dNl[l] = N0;

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
    	//Run the estimation at desired accuracy.

        mcqmc06_l(l,dNl[l],sums);


        suml[0][l] += (float) dNl[l];
        suml[1][l] += sums[1];
        suml[2][l] += sums[2];
        NlCl[l]    += sums[0];
        // sum total cost. cost is given as exponential in level number.
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
      //Expectation at level l : sum(paths of Y) / number of paths in Y
      ml[l] = fabs(suml[1][l]/suml[0][l]);

      //Variance at level l :
      //	sum (paths of Y sq) / num paths
      //	- Expectation squared. if -ve, set to 0.
      Vl[l] = fmaxf(suml[2][l]/suml[0][l] - ml[l]*ml[l], 0.0f);

	    printf("level %d: variance %.5f expectation %.5f\n", l, Vl[l], ml[l]);
      // If gamma 0 - estimate as average cost.
      if (gamma_0 <= 0.0f) Cl[l] = NlCl[l] / suml[0][l];

      //estimates for m_l and v_l not allowed to decrease by more than half
      //of anticipated value.
      if (l>1) {
        ml[l] = fmaxf(ml[l],  0.5f*ml[l-1]/powf(2.0f,alpha));
        Vl[l] = fmaxf(Vl[l],  0.5f*Vl[l-1]/powf(2.0f,beta));
      }

      //Thm 1.1 : C = eps ^ - 2 * [sum of sqrt(v_l * C_l)] ^ 2
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
	      printf(" level %d - %d", l, dNl[l]);
      }
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

      //Convergence test
      if (rem > sqrtf(theta)*eps) {
        if (L==Lmax)
          printf("*** failed to achieve weak convergence *** \n");
        else {

          //Introduce new level.
          converged = 0;
          L++;

          //Estimate new variance and cost for next level
          Vl[L] = Vl[L-1]/powf(2.0f,beta);
          Cl[L] = Cl[L-1]*powf(2.0f,gamma);

          if (diag) printf(" L = %d \n",L);

          sum = 0.0f;
          for (int l=0; l<=L; l++) sum += sqrtf(Vl[l]*Cl[l]);
          for (int l=0; l<=L; l++)
            dNl[l] = ceilf( fmaxf( 0.0f, 
                            sqrtf(Vl[l]/Cl[l])*sum/((1.0f-theta)*eps*eps)
                          - suml[0][l] ) );
        }
      }
    }
  }

  //
  // finally, evaluate multilevel estimator and set outputs
  //

  float P = 0.0f;
  for (int l=0; l<=L; l++) {
	//Average each pay off at each level.
    P    += suml[1][l]/suml[0][l];
    Nl[l] = suml[0][l];
    Cl[l] = NlCl[l] / Nl[l];
  }

  return P;
}



//
// linear regression routine
//

// void regression(int N, float *x, float *y, float &a, float &b){

//   float sum0=0.0f, sum1=0.0f, sum2=0.0f, sumy0=0.0f, sumy1=0.0f;

//   for (int i=0; i<N; i++) {
//     sum0  += 1.0f;
//     sum1  += x[i];
//     sum2  += x[i]*x[i];

//     sumy0 += y[i];
//     sumy1 += y[i]*x[i];
//   }

//   a = (sum0*sumy1 - sum1*sumy0) / (sum0*sum2 - sum1*sum1);
//   b = (sum2*sumy0 - sum1*sumy1) / (sum0*sum2 - sum1*sum1);
// }
