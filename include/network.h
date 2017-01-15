#ifndef NETWORK_H
#define NETWORK_H

#include <algorithm>
#include <cstring>
#include <iostream>

#include "layer.h"

struct Network {
  Layer **layers;

  unsigned int count_layers;

  void init_network(unsigned int *inputs, unsigned int *neurons,
                    unsigned int clayers);

  void propagate_network(const double *input);

  double train_network(const double *input, const double *awaited_output,
                       const double learning_rate, double momentum);

  ~Network();
};

#endif
