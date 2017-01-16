#include "network.h"

void Network::init_network(unsigned int *inputs, unsigned int *neurons,
                           unsigned int clayers) {

  count_layers = clayers;

  layers = new Layer *[clayers];

  for (unsigned int l = 0; l < clayers; l++) {
    layers[l] = new Layer;
    layers[l]->init_layer(inputs[l], neurons[l]);
  }
}

void Network::propagate_network(const float *input) {
  std::memcpy(layers[0]->input, input, layers[0]->count_input * sizeof(float));

  for (unsigned int l = 0; l < count_layers; l++) {
    layers[l]->propagate_layer();
    if (l < count_layers - 1) {
      for (unsigned int n = 0; n < layers[l]->count_neurons; n++) {
        layers[l + 1]->input[n] = layers[l]->neurons[n]->output;
      }
    }
  }
}

float Network::train_network(const float *input, const float *awaited_output,
                              const float learning_rate, float momentum) {

  propagate_network(input);
  float total_error = 0;
  float out, delta, dw;

  Layer *output_layer = layers[count_layers - 1];

  for (unsigned int i = 0; i < output_layer->count_neurons; i++) {
    out = output_layer->neurons[i]->output;

    total_error += 0.5 * (awaited_output[i] - out) * (awaited_output[i] - out);

    output_layer->neurons[i]->delta =
        (awaited_output[i] - out) * out * (1 - out);
    /*
        std::cout << " " << i << " | " << out << " " << awaited_output[i] << " |
       "
                  << (awaited_output[i] - out) * out * (1 - out) << std::endl;
       */

    output_layer->neurons[i]->wbias +=
        learning_rate * (awaited_output[i] - out) * out * (1 - out);
  }

  for (int l = (int)count_layers - 2; l >= 0; l--) {
    delta = 0;
    for (unsigned int i = 0; i < layers[l + 1]->count_neurons; i++) {
      for (unsigned int j = 0; j < layers[l + 1]->count_input; j++) {
        delta += layers[l + 1]->neurons[i]->weights[j] *
                 layers[l + 1]->neurons[i]->delta;
      }
    }
    for (unsigned int n = 0; n < layers[l]->count_neurons; n++) {
      out = layers[l]->neurons[n]->output;
      /*  std::cout << " " << l << " " << n << " "
                  << " | " << out << " " << delta << " | "
                  << out * (1 - out) * delta << std::endl; */
      layers[l]->neurons[n]->delta = out * (1 - out) * delta;
      layers[l]->neurons[n]->wbias += learning_rate * out * (1 - out) * delta;
    }
  }

  for (unsigned int l = 0; l < count_layers; l++) {
    for (unsigned int n = 0; n < layers[l]->count_neurons; n++) {
      for (unsigned int i = 0; i < layers[l]->count_input; i++) {
        dw = learning_rate * layers[l]->input[i] * layers[l]->neurons[n]->delta;
        dw += momentum * layers[l]->neurons[n]->prvdeltas[i];
        layers[l]->neurons[n]->prvdeltas[i] = dw;
        layers[l]->neurons[n]->weights[i] += dw;
        /*std::cout << " " << l << " " << n << " " << i << " | " << dw << " "
                  << layers[l]->neurons[n]->weights[i] << std::endl;*/
      }
    }
  }
  return total_error;
}

Network::~Network() {
  for (unsigned int i = 0; i < count_layers; i++) {
    delete layers[i];
  }
  delete layers;
}
