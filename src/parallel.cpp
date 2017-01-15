#include "parallel.h"

void Parallel::tile_propagate_layer(Layer *l, int neuron_start,
                                    int neuron_end) {

  double output;

  for (unsigned int i = neuron_start; i < neuron_end; i++) {

    output = 0;

    for (unsigned int j = 0; j < l->count_input; j++) {
      output += l->neurons[i]->weights[j] * l->input[j];
    }

    output += l->neurons[i]->wbias;

    l->neurons[i]->output = 1 / (1 + exp(-output));
  }
}
void Parallel::tile_layer_train(Layer *l, Layer *pl, int neuron_start,
                                int neuron_end, double learning_rate) {
  double out;
  double delta = 0;
  for (unsigned int i = 0; i < pl->count_neurons; i++) {
    for (unsigned int j = neuron_start; j < neuron_end; j++) {
      delta += pl->neurons[i]->weights[j] * pl->neurons[i]->delta;
    }
  }
  for (unsigned int n = neuron_start; n < neuron_end; n++) {
    out = l->neurons[n]->output;

    l->neurons[n]->delta = out * (1 - out) * delta;
    l->neurons[n]->wbias += learning_rate * out * (1 - out) * delta;
  }
}

void Parallel::tile_layer_update(Layer *l, int neuron_start, int neuron_end,
                                 double learning_rate, double momentum) {
  double dw;
  for (unsigned int n = neuron_start; n < neuron_end; n++) {
    for (unsigned int i = 0; i < l->count_input; i++) {
      dw = learning_rate * l->input[i] * l->neurons[n]->delta;
      dw += momentum * l->neurons[n]->prvdeltas[i];
      l->neurons[n]->prvdeltas[i] = dw;
      l->neurons[n]->weights[i] += dw;
    }
  }
}
