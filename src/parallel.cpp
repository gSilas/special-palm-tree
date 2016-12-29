#include "parallel.h"

void Parallel::tile_propagate_layer(Layer *l, int neuron_start) {

  float output;

  for (unsigned int i = neuron_start; i < l->count_neurons; i++) {

    output = 0.f;

    for (unsigned int j = 0; j < l->count_input; j++) {
      output += l->neurons[i]->weights[j] * l->input[j];
    }

    output += l->neurons[i]->wbias;

    l->neurons[i]->output = 1.f / (1.f + exp(-output));
  }
}
