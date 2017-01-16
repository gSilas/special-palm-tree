#ifndef PARALLELNETWORK_H
#define PARALLELNETWORK_H

#include <algorithm>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

#include "layer.h"
#include "parallel.h"

struct ParallelNetwork {
  Layer **layers;

  unsigned int count_layers;

  void init_network(unsigned int *inputs, unsigned int *neurons,
                    unsigned int clayers);

  void propagate_network(const float *input);

  float train_network(const float *input, const float *awaited_output,
                       const float learning_rate, float momentum);

  ~ParallelNetwork();
};

#endif
