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
                                  float *device_delta, float *device_prvdeltas,
                                  float learning_rate, float momentum,
                                  unsigned int input_offset,
                                  unsigned int neuron_offset,
                                  unsigned int input_size,
                                  unsigned int weight_offset);

__global__ void tile_propagate_inlayer(
    float *device_dataset, float *device_input, float *device_weights,
    float *device_wbias, float *device_output, unsigned int input_size,
    unsigned int neuron_size, unsigned int nl_neuron_offset);

__global__ void
tile_propagate_layer(float *device_input, float *device_weights,
                     float *device_wbias, float *device_output,
                     unsigned int input_size, unsigned int neuron_size,
                     unsigned int input_offset, unsigned int neuron_offset,
                     unsigned int nl_neuron_offset, unsigned int weight_offset);

__global__ void tile_outlayer_train(float *device_delta, float *device_wbias,
                                    float *device_output,
                                    float *device_awaited_output,
                                    float learning_rate,
                                    unsigned int nl_neuron_offset);
__global__ void
tile_layer_train(float *device_weights, float *device_wbias,
                 float *device_delta, float *device_output,
                 float *device_awaited_output, float learning_rate,
                 unsigned int pl_neuron_size, unsigned int pl_input_size,
                 unsigned int tl_weight_offset, unsigned int tl_neuron_offset,
                 unsigned int nl_neuron_offset);
}

struct GPUNetwork {

  // connection_land
  int count_layers;
  int *num_blocks;
  int *threads_per_block;
  unsigned int *arr_input_size;
  unsigned int *arr_neuron_size;
  unsigned int *sum_weight_size;
  unsigned int *sum_input_size;
  unsigned int *sum_neuron_size;

  // device_land
  float *device_input;
  float *device_awaited_output;

  float *device_weights;
  float *device_wbias;

  float *device_delta;
  float *device_prvdeltas;
  float *device_output;

  float *device_dataset;
  float *test_device_dataset;
  float *device_labels;

  void init_network(unsigned int *inputs, unsigned int *neurons,
                    unsigned int clayers);

  unsigned int propagate_network(float *data_set, float *label_set,
                                 unsigned int dataset_count, size_t set_size,
                                 size_t label_size);

  void train_network(float *data_set, size_t set_size, float *data_labels,
                     size_t label_size, unsigned int dataset_count,
                     unsigned int epochs, const float learning_rate,
                     float momentum);

  ~GPUNetwork();

  float *getOutput();
};

#endif
