#ifndef NEURON_H
#define NEURON_H

#include <cstdlib>

struct Neuron {

  float *weights;
  float delta;
  float *prvdeltas;
  float output;
  float wbias;

  void init_neuron(unsigned int inputsize);

  ~Neuron();
};

#endif
