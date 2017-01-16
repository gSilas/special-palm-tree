#ifndef CUDAGPUNETWORK_H
#define CUDAGPUNETWORK_H

#include <cuda.h>
#include <cuda_runtime.h>

// include, project
#include "cuda_util.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line/*, bool abort=true*/)
{
     if (code != cudaSuccess)
     {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        //if (abort) exit(code);
     }
}

#endif
