#ifndef LAYER_H
#define LAYER_H

#include <cmath>
#include <iostream>

#include "neuron.h"

struct Layer {

  Neuron **neurons;

  double *input;

  unsigned int count_neurons;
  unsigned int count_input;

  void init_layer(unsigned int insize, unsigned int neuronsize);
  void propagate_layer();

  ~Layer();
};

#endif
