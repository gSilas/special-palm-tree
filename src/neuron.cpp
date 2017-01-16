
#include "neuron.h"

void Neuron::init_neuron(unsigned int inputsize) {
  output = -0.5 +
           static_cast<float>(rand()) /
               (static_cast<float>(RAND_MAX / (0.5 - (-0.5))));
  delta = 0.0;
  wbias = -0.5 +
          static_cast<float>(rand()) /
              (static_cast<float>(RAND_MAX / (0.5 - (-0.5))));
  weights = float[inputsize];
  prvdeltas = float[inputsize];

  for (unsigned int i = 0; i < inputsize; i++) {
    weights[i] =
        0; // static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    prvdeltas[i] = 0;
  }
}

Neuron::~Neuron() {}
