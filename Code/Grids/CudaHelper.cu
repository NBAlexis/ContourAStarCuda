#include "CudaHelper.h"

#pragma region cuda

__global__ void
LAUNCH_BOUND
_kernelReduceComp(cuDoubleComplex* arr, unsigned int uiJump, unsigned int uiMax)
{
    unsigned int uiIdFrom = (threadIdx.x + blockIdx.x * blockDim.x) * (uiJump << 1U) + uiJump;
    if (uiIdFrom < uiMax)
    {
        arr[uiIdFrom - uiJump] = cuCadd(arr[uiIdFrom - uiJump], arr[uiIdFrom]);
    }
}

#pragma endregion

static inline unsigned int GetReduceDim(unsigned int uiLength)
{
    unsigned int iRet = 0;
    while ((1U << iRet) < uiLength)
    {
        ++iRet;
    }
    return iRet;
}

cuDoubleComplex ReduceComplex(cuDoubleComplex* deviceBuffer, unsigned int uiLength)
{
    const unsigned int iRequiredDim = (uiLength + 1U) >> 1U;
    const unsigned int iPower = GetReduceDim(iRequiredDim);
    for (unsigned int i = 0; i <= iPower; ++i)
    {
        unsigned int iJump = 1U << i;
        unsigned int iThreadNeeded = 1U << (iPower - i);
        unsigned int iBlock = iThreadNeeded > _kMaxCudaThread ? iThreadNeeded / _kMaxCudaThread : 1U;
        unsigned int iThread = iThreadNeeded > _kMaxCudaThread ? _kMaxCudaThread : iThreadNeeded;
        _kernelReduceComp <<<iBlock, iThread >>> (deviceBuffer, iJump, uiLength);
    }
    cuDoubleComplex result[1];
    cudaMemcpy(result, deviceBuffer, sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    return result[0];
}
