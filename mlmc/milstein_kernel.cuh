#pragma once
#include "cuda_headers.cuh"

const double K = 100.0;
const double T = 1.0;
const double r = 0.05;
const double sig = 0.2;
const double B = 0.85 * K;

template<int N>
__global__ void milstein_half(float *d_unif, float *d_norm, double *d_v, double *d_v_sq){
    int nf, nc;
    __half hf, hc, X0, Xf, Xc,
	Xf0, Xc0, Xc1, vf, vc, dWc, ddW, Pf, Pc, dP;

    __half dWf[2], dIf[2], Lf[2];
    __half zeroh = __float2half(0.0f);
    __half rh = __float2half(r);
    __half sigh = __float2half(sig);

    //Number of samples of each distribution needed per iteration
    int num_uniform = 2;
    int num_norm = 4;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    int norm_id = threadIdx.x + num_norm * N * blockIdx.x * blockDim.x;
    int unif_id = threadIdx.x + num_uniform * N * blockIdx.x * blockDim.x;

    nf = N;
    nc = N;

    hf = hdiv(__float2half((float)T),__float2half((float) nf));
    hc = hdiv(__float2half((float)T),__float2half((float) nc));

    X0 = __float2half((float)K);
    Xf = X0;
    Xc = Xf;

    //One path timestep
    for (int n=0; n<nc; n++) {
	dWf[0] = __hmul(hsqrt(hf), __float2half(d_norm[norm_id]));
		norm_id += blockDim.x;
		dWf[1] = __hmul(hsqrt(hf), __float2half(d_norm[norm_id]));
		norm_id += blockDim.x;

		Lf[0] = hlog(__float2half(d_unif[unif_id]));
		unif_id += blockDim.x;
		Lf[1] = hlog(__float2half(d_unif[unif_id]));
		unif_id += blockDim.x;

		dIf[0] = __hmul(hsqrt(hdiv(hf,__float2half(12.0f))),
				__hmul(hf,__float2half(d_norm[norm_id])));
		norm_id += blockDim.x;

		dIf[1] = __hmul(hsqrt(hdiv(hf,__float2half(12.0f))),
				__hmul(hf,__float2half(d_norm[norm_id])));
		norm_id += blockDim.x;

		for (int m=0; m<2; m++) {
			Xf0 = Xf;
			__half v1 = __hmul(__hmul(rh, Xf), hf);
			__half v2 = __hmul(__hmul(sigh, Xf), dWf[m]);
			__half p1 = __hmul(__hmul(__float2half(0.5f), __hmul(rh, Xf)), hf);
			__half p2 = __hsub(__hmul(dWf[m], dWf[m]), hf);
			Xf = __hadd(Xf, __hadd(v1, __hadd(v2, __hmul(p1, p2) ) ) );
//			Xf  = Xf + r*Xf*hf + sig*Xf*dWf[m]
//			+ 0.5f*sig*sig*Xf*(dWf[m]*dWf[m]-hf);
			vf  = __hmul(sigh,Xf0);
		}

		dWc = __hadd(dWf[0], dWf[1]);
		ddW = __hsub(dWf[0], dWf[1]);

		Xc0 = Xc;
		__half v1 = __hmul(rh, __hmul(Xc, hc));
		__half v2 = __hmul(sigh, __hmul(Xc, dWc));
	        __half p1 = __hmul(__hmul(__float2half(0.5f), __hmul(sigh, sigh)), Xc);
		__half p2 = __hsub(__hmul(dWc, dWc), hc);
//		Xc  = Xc + r*Xc*hc + sig*Xc*dWc + 0.5f*sig*sig*Xc*(dWc*dWc-hc);

		vc  = __hmul(sigh,Xc0);
		Xc1 = __hmul(__float2half(0.5f), __hadd(Xc0, __hadd(Xc, __hmul(vc, ddW))));
//		Xc1 = 0.5f*(Xc0 + Xc + vc*ddW);
			   
    }
		  
    Pf  = __hsub(Xf, __float2half((float)K));
    if (__hlt(Pf,zeroh)) Pf = zeroh;
    Pc  = __hsub(Xc, __float2half((float)K));
    if (__hlt(Pc,zeroh)) Pc = zeroh;

    dP  = __hmul(hexp(__hmul(__float2half((float)(-r)),__float2half(T))),__hsub(Pf,Pc));

    printf("dp %g pf %g pc %g VAL : %g \n ", dP, Pf, Pc, __hsub(Pf,Pc));
    
    d_v[tid] = (double)__half2float(dP);
    d_v_sq[tid] = (double)__half2float(__hmul(dP,dP));
}

template<int N>
__global__ void milstein_float(float *d_unif, float *d_norm, double *d_v, double *d_v_sq){
    int   M, nf, nc;
    float sig, B, hf, hc, X0, Xf, Xc,
	Xf0, Xc0, Xc1, vf, vc, dWc, ddW, Pf, Pc, dP, K;

    float dWf[2], dIf[2], Lf[2];

    //Number of samples of each distribution needed per iteration
    int num_uniform = 2;
    int num_norm = 4;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    int norm_id = threadIdx.x + num_norm * N * blockIdx.x * blockDim.x;
    int unif_id = threadIdx.x + num_uniform * N * blockIdx.x * blockDim.x;

    nf = N;
    nc = N;

    hf = T / ((float) nf);
    hc = T / ((float) nc);

    X0 = K;
    Xf = X0;
    Xc = Xf;

    //One path timestep
    for (int n=0; n<nc; n++) {
		dWf[0] = sqrt(hf)*d_norm[norm_id];
		norm_id += blockDim.x;
		dWf[1] = sqrt(hf)*d_norm[norm_id];
		norm_id += blockDim.x;

		Lf[0] = logf(d_unif[unif_id]);
		unif_id += blockDim.x;
		Lf[1] = logf(d_unif[unif_id]);
		unif_id += blockDim.x;

		dIf[0] = sqrt(hf/12.0f)*hf*d_norm[norm_id];
		norm_id += blockDim.x;

		dIf[1] = sqrt(hf/12.0f)*hf*d_norm[norm_id];
		norm_id += blockDim.x;

		for (int m=0; m<2; m++) {
			Xf0 = Xf;
			Xf  = Xf + r*Xf*hf + sig*Xf*dWf[m]
			+ 0.5f*sig*sig*Xf*(dWf[m]*dWf[m]-hf);
			vf  = sig*Xf0;
		}

		dWc = dWf[0] + dWf[1];
		ddW = dWf[0] - dWf[1];

		Xc0 = Xc;
		Xc  = Xc + r*Xc*hc + sig*Xc*dWc + 0.5f*sig*sig*Xc*(dWc*dWc-hc);

		vc  = sig*Xc0;
		Xc1 = 0.5f*(Xc0 + Xc + vc*ddW);
			   
    }

    Pf  = fmaxf(0.0f,Xf-K);
    Pc  = fmaxf(0.0f,Xc-K);

    dP  = exp(-r*T)*(Pf-Pc);

    printf("dp %g pf %g pc %g VAL : %g \n ", dP, Pf, Pc, Pf - Pc);
    
    d_v[tid] = dP;
    d_v_sq[tid] = dP*dP;

    // HALF Calc
    {
    __half hf, hc, X0, Xf, Xc,
	Xf0, Xc0, Xc1, vf, vc, dWc, ddW, Pf, Pc, dP;

    __half dWf[2], dIf[2], Lf[2];
    __half zeroh = __float2half(0.0f);
    __half rh = __float2half(r);
    __half sigh = __float2half(sig);

    //Number of samples of each distribution needed per iteration
    int num_uniform = 2;
    int num_norm = 4;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    int norm_id = threadIdx.x + num_norm * N * blockIdx.x * blockDim.x;
    int unif_id = threadIdx.x + num_uniform * N * blockIdx.x * blockDim.x;

    nf = N;
    nc = N;

    hf = hdiv(__float2half((float)T),__float2half((float) nf));
    hc = hdiv(__float2half((float)T),__float2half((float) nc));

    X0 = __float2half((float)K);
    Xf = X0;
    Xc = Xf;

    //One path timestep
    for (int n=0; n<nc; n++) {
	dWf[0] = __hmul(hsqrt(hf), __float2half(d_norm[norm_id]));
		norm_id += blockDim.x;
		dWf[1] = __hmul(hsqrt(hf), __float2half(d_norm[norm_id]));
		norm_id += blockDim.x;

		Lf[0] = hlog(__float2half(d_unif[unif_id]));
		unif_id += blockDim.x;
		Lf[1] = hlog(__float2half(d_unif[unif_id]));
		unif_id += blockDim.x;

		dIf[0] = __hmul(hsqrt(hdiv(hf,__float2half(12.0f))),
				__hmul(hf,__float2half(d_norm[norm_id])));
		norm_id += blockDim.x;

		dIf[1] = __hmul(hsqrt(hdiv(hf,__float2half(12.0f))),
				__hmul(hf,__float2half(d_norm[norm_id])));
		norm_id += blockDim.x;

		for (int m=0; m<2; m++) {
			Xf0 = Xf;
			__half v1 = __hmul(__hmul(rh, Xf), hf);
			__half v2 = __hmul(__hmul(sigh, Xf), dWf[m]);
			__half p1 = __hmul(__hmul(__float2half(0.5f), __hmul(rh, Xf)), hf);
			__half p2 = __hsub(__hmul(dWf[m], dWf[m]), hf);
			Xf = __hadd(Xf, __hadd(v1, __hadd(v2, __hmul(p1, p2) ) ) );
//			Xf  = Xf + r*Xf*hf + sig*Xf*dWf[m]
//			+ 0.5f*sig*sig*Xf*(dWf[m]*dWf[m]-hf);
			vf  = __hmul(sigh,Xf0);
		}

		dWc = __hadd(dWf[0], dWf[1]);
		ddW = __hsub(dWf[0], dWf[1]);

		Xc0 = Xc;
		__half v1 = __hmul(rh, __hmul(Xc, hc));
		__half v2 = __hmul(sigh, __hmul(Xc, dWc));
	        __half p1 = __hmul(__hmul(__float2half(0.5f), __hmul(sigh, sigh)), Xc);
		__half p2 = __hsub(__hmul(dWc, dWc), hc);
//		Xc  = Xc + r*Xc*hc + sig*Xc*dWc + 0.5f*sig*sig*Xc*(dWc*dWc-hc);

		vc  = __hmul(sigh,Xc0);
		Xc1 = __hmul(__float2half(0.5f), __hadd(Xc0, __hadd(Xc, __hmul(vc, ddW))));
//		Xc1 = 0.5f*(Xc0 + Xc + vc*ddW);
			   
    }
		  
    Pf  = __hsub(Xf, __float2half((float)K));
    if (__hlt(Pf,zeroh)) Pf = zeroh;
    Pc  = __hsub(Xc, __float2half((float)K));
    if (__hlt(Pc,zeroh)) Pc = zeroh;

    dP  = __hmul(hexp(__hmul(__float2half((float)(-r)),__float2half(T))),__hsub(Pf,Pc));

    printf("dp %g pf %g pc %g VAL : %g \n ", dP, Pf, Pc, __hsub(Pf,Pc));
    
    d_v[tid] -= (double)__half2float(dP);
    d_v_sq[tid] -= (double)__half2float(__hmul(dP,dP));
    }
}

template<int N>
__global__ void milstein_double(float *d_unif, float *d_norm, double *d_v, double *d_v_sq){
    int   M, nf, nc;
    double sig, B, hf, hc, X0, Xf, Xc,
	Xf0, Xc0, Xc1, vf, vc, dWc, ddW, Pf, Pc, dP, K;

    double dWf[2], dIf[2], Lf[2];

    //Number of samples of each distribution needed per iteration
    int num_uniform = 2;
    int num_norm = 4;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    int norm_id = threadIdx.x + num_norm * N * blockIdx.x * blockDim.x;
    int unif_id = threadIdx.x + num_uniform * N * blockIdx.x * blockDim.x;

    nf = N;
    nc = N;

    hf = T / ((float) nf);
    hc = T / ((float) nc);

    X0 = K;
    Xf = X0;
    Xc = Xf;

    //One path timestep
    for (int n=0; n<nc; n++) {
		dWf[0] = sqrt(hf)*d_norm[norm_id];
		norm_id += blockDim.x;
		dWf[1] = sqrt(hf)*d_norm[norm_id];
		norm_id += blockDim.x;

		Lf[0] = logf(d_unif[unif_id]);
		unif_id += blockDim.x;
		Lf[1] = logf(d_unif[unif_id]);
		unif_id += blockDim.x;

		dIf[0] = sqrt(hf/12.0f)*hf*d_norm[norm_id];
		norm_id += blockDim.x;

		dIf[1] = sqrt(hf/12.0f)*hf*d_norm[norm_id];
		norm_id += blockDim.x;

		for (int m=0; m<2; m++) {
			Xf0 = Xf;
			Xf  = Xf + r*Xf*hf + sig*Xf*dWf[m]
			+ 0.5f*sig*sig*Xf*(dWf[m]*dWf[m]-hf);
			vf  = sig*Xf0;
		}

		dWc = dWf[0] + dWf[1];
		ddW = dWf[0] - dWf[1];

		Xc0 = Xc;
		Xc  = Xc + r*Xc*hc + sig*Xc*dWc + 0.5f*sig*sig*Xc*(dWc*dWc-hc);

		vc  = sig*Xc0;
		Xc1 = 0.5f*(Xc0 + Xc + vc*ddW);
			   
    }

    Pf  = fmaxf(0.0f,Xf-K);
    Pc  = fmaxf(0.0f,Xc-K);

    dP  = exp(-r*T)*(Pf-Pc);

    printf("dp %g pf %g pc %g VAL : %g \n ", dP, Pf, Pc, Pf - Pc);
    
    d_v[tid] = dP;
    d_v_sq[tid] = dP*dP;

    // Float calculation
    {
        float sig, B, hf, hc, X0, Xf, Xc,
	Xf0, Xc0, Xc1, vf, vc, dWc, ddW, Pf, Pc, dP, K;

    float dWf[2], dIf[2], Lf[2];

    //Number of samples of each distribution needed per iteration
    int num_uniform = 2;
    int num_norm = 4;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    int norm_id = threadIdx.x + num_norm * N * blockIdx.x * blockDim.x;
    int unif_id = threadIdx.x + num_uniform * N * blockIdx.x * blockDim.x;

    nf = N;
    nc = N;

    hf = T / ((float) nf);
    hc = T / ((float) nc);

    X0 = K;
    Xf = X0;
    Xc = Xf;

    //One path timestep
    for (int n=0; n<nc; n++) {
		dWf[0] = sqrt(hf)*d_norm[norm_id];
		norm_id += blockDim.x;
		dWf[1] = sqrt(hf)*d_norm[norm_id];
		norm_id += blockDim.x;

		Lf[0] = logf(d_unif[unif_id]);
		unif_id += blockDim.x;
		Lf[1] = logf(d_unif[unif_id]);
		unif_id += blockDim.x;

		dIf[0] = sqrt(hf/12.0f)*hf*d_norm[norm_id];
		norm_id += blockDim.x;

		dIf[1] = sqrt(hf/12.0f)*hf*d_norm[norm_id];
		norm_id += blockDim.x;

		for (int m=0; m<2; m++) {
			Xf0 = Xf;
			Xf  = Xf + r*Xf*hf + sig*Xf*dWf[m]
			+ 0.5f*sig*sig*Xf*(dWf[m]*dWf[m]-hf);
			vf  = sig*Xf0;
		}

		dWc = dWf[0] + dWf[1];
		ddW = dWf[0] - dWf[1];

		Xc0 = Xc;
		Xc  = Xc + r*Xc*hc + sig*Xc*dWc + 0.5f*sig*sig*Xc*(dWc*dWc-hc);

		vc  = sig*Xc0;
		Xc1 = 0.5f*(Xc0 + Xc + vc*ddW);
			   
    }

    Pf  = fmaxf(0.0f,Xf-K);
    Pc  = fmaxf(0.0f,Xc-K);

    dP  = exp(-r*T)*(Pf-Pc);

    printf("dp %g pf %g pc %g VAL : %g \n ", dP, Pf, Pc, Pf - Pc);
    
    d_v[tid] -= dP;
    d_v_sq[tid] -= dP*dP;
}

}
