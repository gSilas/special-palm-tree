#ifndef GPUNETWORK_H
#define GPUNETWORK_H

#include <algorithm>
#include <cstring>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda_util.h"
#include "device.cuh"

struct GPUNetwork {

  // connection_land
  int count_layers;
  int *num_blocks;
  int *threads_per_block;

  // device_land
  float **device_inputs;
  float *device_output;
  float *device_awaited_output;

  float **device_weights;
  float **device_wbias;

  float **device_delta;
  float **device_prvdeltas;

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
