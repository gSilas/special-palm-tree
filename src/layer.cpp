#include "layer.h"

void Layer::init_layer(unsigned int insize, unsigned int neuronsize) {

  neurons = new Neuron[neuronsize];

  for (unsigned int i = 0; i < neuronsize; i++) {
    neurons[i].init_neuron(insize);
  }

  input = new float[insize];
  count_neurons = neuronsize;
  count_input = insize;
}

void Layer::propagate_layer() {

  float output;

  for (unsigned int i = 0; i < count_neurons; i++) {

    output = 0;

    for (unsigned int j = 0; j < count_input; j++) {
      output += (neurons[i].weights[j] * input[j]);
    }

    output += neurons[i].wbias;

    neurons[i].output = 1 / (1 + exp(-output));
  }
}

Layer::~Layer() {
  delete neurons;
  delete input;
}
