#ifndef NEURON_H
#define NEURON_H

#include <cstdlib>
#include <utility>

class Neuron {
public:
  float *weights;
  float wbias;

  float delta;
  float *prvdeltas;

  float output;

  void init_neuron(unsigned int inputsize);

  ~Neuron();
};

#endif
