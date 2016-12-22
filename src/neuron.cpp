
#include "neuron.h"

void Neuron::init_neuron(unsigned int inputsize) {
  output = 0;
  weights = new float[inputsize];
  deltas = new float[inputsize];

  float sign = -1.0f;
  for (unsigned int i = 0; i < inputsize; i++) {
    weights[i] = (float(rand()) / float(RAND_MAX)) / 2.f * sign;
    deltas[i] = 0.f;
    sign *= -1.0f;
  }
  bias = 1.0f;
  wbias = (float(rand()) / float(RAND_MAX)) / 2.f * (sign * -1.0f);
}

Neuron::~Neuron() {
  delete weights;
  delete deltas;
}
