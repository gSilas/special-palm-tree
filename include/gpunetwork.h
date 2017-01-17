#ifndef GPUNETWORK_H
#define GPUNETWORK_H

#include <algorithm>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

#include "gpunetwork.cuh"
#include "layer.h"
#include "neuron.h"

struct GPUNetwork {

  float *deviceIn;
  Layer **net_layers;
  Layer **device_layers;

  unsigned int count_layers;

  void init_network(unsigned int *inputs, unsigned int *neurons,
                    unsigned int clayers);

  void propagate_network(const float *input);

  float train_network(const float *input, const float *awaited_output,
                      const float learning_rate, float momentum);

  ~GPUNetwork();
};

#endif
