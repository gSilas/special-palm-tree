#ifndef NETWORK_H
#define NETWORK_H

#include <algorithm>
#include <cstring>
#include <iostream>

#include "layer.h"

struct Network {
  Layer **layers;

  unsigned int count_hiddenlayers;

  void init_network(unsigned int *inputs, unsigned int *neurons,
                    unsigned int chidden_layers);

  void propagate_network(const float *input);

  float train_network(const float *input, const float *awaited_output,
                      const float learning_rate, float momentum);

  float *getOutput();

  ~Network();
};

#endif
