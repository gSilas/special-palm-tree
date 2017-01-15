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

  void propagate_network(const double *input);

  double train_network(const double *input, const double *awaited_output,
                       const double learning_rate, double momentum);

  ~ParallelNetwork();
};

#endif
