#pragma once
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
