
#include "neuron.h"

void Neuron::init_neuron(unsigned int inputsize) {
  output = 0.f;
  delta = 0.f;
  wbias = 0.0f;
  weights = new float[inputsize];
  prvdeltas = new float[inputsize];

  for (unsigned int i = 0; i < inputsize; i++) {
    weights[i] = 0.f;
    prvdeltas[i] = 0.f;
  }
}

Neuron::~Neuron() {
  delete weights;
  delete prvdeltas;
}
