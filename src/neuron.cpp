
#include "neuron.h"

void Neuron::init_neuron(unsigned int inputsize) {
  output = -0.5 +
           static_cast<float>(rand()) /
               (static_cast<float>(RAND_MAX / (0.5 - (-0.5))));
  delta = 0.0;
  wbias = -0.5 +
          static_cast<float>(rand()) /
              (static_cast<float>(RAND_MAX / (0.5 - (-0.5))));

  weights = new float[inputsize];
  prvdeltas = new float[inputsize];

  for (unsigned int i = 0; i < inputsize; i++) {
    weights[i] = 0;
    prvdeltas[i] = 0;
  }
}

Neuron::~Neuron() {
  delete weights;
  delete prvdeltas;
}
