#ifndef CUDADEVICE_H
#define CUDADEVICE_H

#include <algorithm>
#include <cstring>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda_util.h"

namespace Device {

__global__ void tile_update_layer(float *device_input, float *device_weights,
                                  float *device_delta, float *device_prvdeltas,
                                  float learning_rate, float momentum,
                                  unsigned int input_size,
                                  unsigned int neuron_size);

__global__ void tile_propagate_inlayer(float *data_set, float *device_input,
                                       float *nl_device_input,
                                       float *device_weights,
                                       float *device_wbias,
                                       unsigned int input_size,
                                       unsigned int neuron_size);

__global__ void tile_propagate_layer(float *device_input,
                                     float *nl_device_input,
                                     float *device_weights, float *device_wbias,
                                     unsigned int input_size,
                                     unsigned int neuron_size);

__global__ void tile_outlayer_train(float *device_output, float *device_delta,
                                    float *device_wbias,
                                    float *device_awaited_output,
                                    float learning_rate);

__global__ void tile_layer_train(float *device_output, float *pl_device_weights,
                                 float *pl_device_delta, float *device_wbias,
                                 float *device_delta,
                                 float *device_awaited_output,
                                 float learning_rate,
                                 unsigned int pl_input_size,
                                 unsigned int pl_neuron_size);
}

#endif
