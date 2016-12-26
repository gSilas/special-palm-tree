#include "network.h"

void Network::init_network(unsigned int *inputs, unsigned int *neurons,
                           unsigned int chidden_layers) {

  count_hiddenlayers = chidden_layers;

  layers = new Layer *[chidden_layers + 2];

  for (unsigned int i = 0; i < chidden_layers + 2; i++) {
    layers[i] = new Layer;
    layers[i]->init_layer(inputs[i], neurons[i]);
  }
}

void Network::propagate_network(const float *input) {
  std::memcpy(layers[0]->input, input, layers[0]->count_input * sizeof(float));

  for (unsigned int i = 0; i < count_hiddenlayers + 2; i++) {
    layers[i]->propagate_layer();
    if (i < count_hiddenlayers + 1) {
      for (unsigned int j = 0; j < layers[i]->count_neurons; j++) {
        layers[i + 1]->input[j] = layers[i]->neurons[j]->output;
      }
    }
  }
}

float *Network::getOutput() {
  float *output = new float[layers[count_hiddenlayers + 1]->count_neurons];
  for (unsigned int i = 0; i < layers[count_hiddenlayers + 1]->count_neurons;
       i++) {
    output[i] = layers[count_hiddenlayers + 1]->neurons[i]->output;
  }
  return output;
}

float Network::train_network(const float *input, const float *awaited_output,
                             const float learning_rate, float momentum) {

  propagate_network(input);
  float total_error = 0.f;
  float out;
  float delta;
  float dw;
  Layer *output_layer = layers[count_hiddenlayers + 1];

  for (unsigned int i = 0; i < output_layer->count_neurons; i++) {
    out = output_layer->neurons[i]->output;

    total_error += 0.5f * (awaited_output[i] - out) * (awaited_output[i] - out);

    output_layer->neurons[i]->delta =
        (awaited_output[i] - out) * out * (1 - out);

    output_layer->neurons[i]->wbias +=
        learning_rate * (awaited_output[i] - out) * out * (1 - out);
    std::cout << "Delta O " << output_layer->neurons[i]->delta << std::endl;
  }

  for (int l = (int)count_hiddenlayers; l > -1; l--) {
    for (unsigned int n = 0; n < layers[l]->count_neurons; n++) {
      out = layers[l]->neurons[n]->output;
      delta = 0.f;
      for (unsigned int i = 0; i < layers[l + 1]->count_neurons; i++) {
        delta += layers[l + 1]->neurons[i]->weights[n] *
                 layers[l + 1]->neurons[i]->delta;
      }
      std::cout << "Delta OL 1 " << layers[l]->neurons[n]->delta << std::endl;
      layers[l]->neurons[n]->delta = out * (1 - out) * delta;
      layers[l]->neurons[n]->wbias += learning_rate * out * (1 - out) * delta;
      std::cout << "Delta OL 2 " << layers[l]->neurons[n]->delta << std::endl;
    }
  }

  for (unsigned int l = 0; l < count_hiddenlayers + 2; l++) {
    for (unsigned int n = 0; n < layers[l]->count_neurons; n++) {
      for (unsigned int i = 0; i < layers[l]->count_input; i++) {
        dw = learning_rate * layers[l]->input[i] * layers[l]->neurons[n]->delta;
        dw += momentum * layers[l]->neurons[n]->prvdeltas[i];
        layers[l]->neurons[n]->prvdeltas[i] = dw;
        layers[l]->neurons[n]->weights[i] += dw;
      }
    }
  }

  return total_error;
}

Network::~Network() {
  for (unsigned int i = 0; i < count_hiddenlayers + 2; i++) {
    delete layers[i];
  }
}
