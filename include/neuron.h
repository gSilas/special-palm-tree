#ifndef NEURON_H
#define NEURON_H

#include <cstdlib>

struct Neuron {

  float *weights;
  float *deltas;
  float output;
  float bias;
  float wbias;

  void init_neuron(unsigned int inputsize);

  ~Neuron();
};

#endif
