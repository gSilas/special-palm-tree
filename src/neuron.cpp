
#include "neuron.h"

void Neuron::init_neuron(unsigned int inputsize) {
  output = -0.5 +
           static_cast<double>(rand()) /
               (static_cast<double>(RAND_MAX / (0.5 - (-0.5))));
  delta = 0.0;
  wbias = -0.5 +
          static_cast<double>(rand()) /
              (static_cast<double>(RAND_MAX / (0.5 - (-0.5))));
  weights = new double[inputsize];
  prvdeltas = new double[inputsize];

  for (unsigned int i = 0; i < inputsize; i++) {
    weights[i] =
        0; // static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
    prvdeltas[i] = 0;
  }
}

Neuron::~Neuron() {
  delete weights;
  delete prvdeltas;
}
