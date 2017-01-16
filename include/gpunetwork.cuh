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
namespace Device {
/*
__global__
void tile_layer_update(Layer *l, int neuron_start, int neuron_end, float learning_rate, float momentum);

__global__
void tile_propagate_layer(Layer *l, int neuron_start, int neuron_end);

__global__
void tile_layer_train(Layer *l, Layer *pl, int neuron_start, int neuron_end, float learning_rate);
*/
}


#endif
