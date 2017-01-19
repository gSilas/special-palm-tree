#ifndef GPUNETWORK_H
#define GPUNETWORK_H

#include <algorithm>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda_util.h"
#include "layer.h"
#include "neuron.h"

#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file,
                      int line /*, bool abort=true*/) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    // if (abort) exit(code);
  }
}

namespace Device {

__global__ void tile_update_layer(float *device_input, float *device_weights,
                                  unsigned int input_offset,
                                  unsigned int neuron_offset,
                                  float learning_rate, float momentum,
                                  float *device_delta, float *device_prvdeltas);

__global__ void tile_propagate_layer(float *device_input, float *device_weights,
                                     float *device_wbias, float *device_output,
                                     unsigned int input_size,
                                     unsigned int neuron_size,
                                     unsigned int input_offset,
                                     unsigned int neuron_offset);
__global__ void tile_outlayer_train(float *device_wbias, float *device_output,
                                    float *device_awaited_output,
                                    unsigned int neuron_offset, float momentum,
                                    float *device_delta);
__global__ void tile_layer_train(float* device_weights,float *device_wbias,
                                         float *device_output,
                                         float *device_awaited_output,
                                         unsigned int neuron_offset,
                                         float learning_rate, float *device_delta,
                                         unsigned int layer_offset);
}

struct GPUNetwork {

  // connection_land
  int count_layers;
  int *num_blocks;
  int *threads_per_block;
  unsigned int *input_size;
  unsigned int *neuron_size;
  unsigned int *sum_input_size;
  unsigned int *sum_neuron_size;

  // device_land
  float *device_input;

  float *device_weights;
  float *device_wbias;

  float *device_delta;
  float *device_prvdeltas;
  float *device_output;

  void init_network(unsigned int *inputs, unsigned int *neurons,
                    unsigned int clayers);

  void propagate_network(const float *input);

  float train_network(const float *input, const float *awaited_output,
                      const float learning_rate, float momentum);

  ~GPUNetwork();

  float *getOutput();
};

#endif
