#include "gpunetwork.h"

void GPUNetwork::init_network(unsigned int *inputs, unsigned int *neurons,
                              unsigned int clayers) {

  count_layers = clayers;

  net_inputs = inputs;

  int neuron_count;

  for (unsigned int l = 0; l < clayers; l++) {
    gpuErrchk(cudaMalloc(&net_inputs[l], sizeof(float) * inputs[l]));
    neuron_count += neurons[l];
  }
  gpuErrchk(cudaMalloc(&net_neurons, sizeof(Neuron) * neuron_count));
}

void GPUNetwork::propagate_network(const float *input) {

  gpuErrchk(cudaMemcpy(&net_inputs[0], &input, sizeof(float) * inputs[i],
                       cudaMemcpyHostToDevice));

  for (unsigned int l = 1; l < count_layers; l++) {

    Device::tile_propagate_layer(net_inputs[l - 1], net_inputs[l],
                                 layers[l]->count_neurons);

    if (l < count_layers - 1) {
      for (unsigned int n = 0; n < layers[l]->count_neurons; n++) {
        layers[l + 1]->input[n] = layers[l]->neurons[n]->output;
      }
    }
  }
}

float GPUNetwork::train_network(const float *input, const float *awaited_output,
                                const float learning_rate, float momentum) {

  propagate_network(input);

  float total_error = 0;
  float out;

  Layer *output_layer = layers[count_layers - 1];

  for (unsigned int i = 0; i < output_layer->count_neurons; i++) {
    out = output_layer->neurons[i]->output;

    total_error += 0.5 * (awaited_output[i] - out) * (awaited_output[i] - out);

    output_layer->neurons[i]->delta =
        (awaited_output[i] - out) * out * (1 - out);

    output_layer->neurons[i]->wbias +=
        learning_rate * (awaited_output[i] - out) * out * (1 - out);
  }

  Device::tile_layer_train(layers[l], layers[l + 1], tiling[i], tiling[i + 1],
                           learning_rate);

  for (unsigned int l = 0; l < count_layers; l++) {
    Device::tile_layer_update(layers[l], tiling[i], tiling[i + 1],
                              learning_rate, momentum);
  }

  return total_error;
}

GPUNetwork::~GPUNetwork() {
  for (unsigned int i = 0; i < count_layers; i++) {
    delete layers[i];
  }
  delete layers;
}
