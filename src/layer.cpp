#include "layer.h"

void Layer::init_layer(unsigned int insize, unsigned int neuronsize) {
  neurons = new Neuron *[neuronsize];
  input = new float[insize];

  for (unsigned int i = 0; i < neuronsize; i++) {
    neurons[i] = new Neuron;
    neurons[i]->init_neuron(insize);
  }

  count_neurons = neuronsize;
  count_input = insize;
}

void Layer::propagate_layer() {

  float output;

  for (unsigned int i = 0; i < count_neurons; i++) {

    output = 0;

    for (unsigned int j = 0; j < count_input; j++) {
      output += (neurons[i]->weights[j] * input[j]);
      /*std::cout << i << " " << j << " | " << neurons[i]->weights[j] << " | "
                << input[j] << " " << output << std::endl;*/
    }

    output += neurons[i]->wbias;

    neurons[i]->output = 1 / (1 + exp(-output));
    /*
        std::cout << i << " | " << output << " | " << 1 / (1 + exp(output))
                  << std::endl; */
  }
}

Layer::~Layer() {

  for (unsigned int i = 0; i < count_neurons; i++) {
    delete neurons[i];
  }
  delete neurons;
  delete input;
}
