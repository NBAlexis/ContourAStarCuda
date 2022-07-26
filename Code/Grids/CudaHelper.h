#pragma once

#include <stdio.h>
#include <string>
#include "cuda_runtime.h"
#include "vector_types.h"
#include "cuda.h"
#include "cuComplex.h"

const unsigned int _kMaxCudaThread = 256U;

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

#ifdef __DRIVER_TYPES_H__
static const char *_cudaGetErrorEnum(cudaError_t error) 
{
    return cudaGetErrorName(error);
}
#endif

#ifdef __DRIVER_TYPES_H__
#ifndef DEVICE_RESET
#define DEVICE_RESET cudaDeviceReset();
#endif
#else
#ifndef DEVICE_RESET
#define DEVICE_RESET
#endif
#endif

#if DEBUG
#define LAUNCH_BOUND 
#else
#define LAUNCH_BOUND __launch_bounds__(_kMaxCudaThread, 1)
#endif

template <typename T> void check(T result, char const *const func, const char *const file, int const line) 
{
    if (result) 
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
        DEVICE_RESET
            // Make sure we call CUDA Device Reset before exiting
            exit(EXIT_FAILURE);
    }
}

extern cuDoubleComplex ReduceComplex(cuDoubleComplex* deviceBuffer, unsigned int uiLength);

__host__ __device__ static __inline__ cuDoubleComplex cuCmulcr(cuDoubleComplex x, double y)
{
    return make_cuDoubleComplex(cuCreal(x) * y, cuCimag(x) * y);
}


