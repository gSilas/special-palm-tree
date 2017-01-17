#ifndef LAYER_H
#define LAYER_H

#include <cmath>
#include <iostream>
#include <utility>

#include "neuron.h"

class Layer {
public:
  Neuron *neurons;

  float *input;

  unsigned int count_neurons;
  unsigned int count_input;

  void init_layer(unsigned int insize, unsigned int neuronsize);
  void propagate_layer();

  ~Layer();
};

#endif
