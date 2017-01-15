#ifndef NEURON_H
#define NEURON_H

#include <cstdlib>

struct Neuron {

  double *weights;
  double wbias;

  double delta;
  double *prvdeltas;

  double output;

  void init_neuron(unsigned int inputsize);

  ~Neuron();
};

#endif
