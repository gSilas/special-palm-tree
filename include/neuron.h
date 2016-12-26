#ifndef NEURON_H
#define NEURON_H

#include <cstdlib>

struct Neuron {

  float *weights;
  float wbias;

  float delta;
  float *prvdeltas;

  float output;

  void init_neuron(unsigned int inputsize);

  ~Neuron();
};

#endif
