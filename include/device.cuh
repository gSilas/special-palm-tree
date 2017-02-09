#ifndef CUDADEVICE_H
#define CUDADEVICE_H

#include <algorithm>
#include <cstring>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda_util.h"

namespace Device {
__global__ void set_dataset(float *device_input, float *data_set,
                            unsigned int input_size);

__global__ void set_layer_memory(float *device_delta, float *device_prvdeltas,
                                 unsigned int input_size,
                                 unsigned int neuron_size);

__global__ void neuron_update_layer(float *device_input, float *device_weights,
                                  float *device_delta, float *device_prvdeltas,
                                  float learning_rate, float momentum,
                                  unsigned int input_size,
                                  unsigned int neuron_size);


__global__ void
neuron_propagate_inlayer(float *device_input, float *nl_device_input,
                       float *device_weights, float *device_wbias,
                       unsigned int input_size, unsigned int neuron_size);

__global__ void reduction(float *data, float* out_data, unsigned int size);

__global__ void neuron_layer_delta(float *device_delta_summands,
                                         float *pl_device_weights,
                                         float *pl_device_delta,
                                         unsigned int input_size,
                                         unsigned int neuron_size);

__global__ void neuron_propagate_layer(float *device_input,
                                     float *nl_device_input,
                                     float *device_weights, float *device_wbias,
                                     unsigned int input_size,
                                     unsigned int neuron_size);

__global__ void neuron_outlayer_train(float *device_output, float *device_delta,
                                    float *device_wbias,
                                    float *device_awaited_output,
                                    float learning_rate,
                                    unsigned int input_size,
                                    unsigned int neuron_size);

__global__ void
neuron_layer_train(float *device_output,float* device_delta_summands, float *device_wbias,
                 float *device_delta, float *device_awaited_output,
                 float learning_rate, unsigned int pl_input_size,
                 unsigned int pl_neuron_size, unsigned int input_size,
                 unsigned int neuron_size);
}

#endif
