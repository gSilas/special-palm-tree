#include "network.h"

void Network::init_network(unsigned int input_size, unsigned int input_neurons,
                           unsigned int output_size,
                           unsigned int *hidden_inputs,
                           unsigned int chidden_layers) {
  if (chidden_layers > 0) {
    hidden_layers = new Layer *[chidden_layers];
  }
  count_hiddenlayers = chidden_layers;

  input_layer = new Layer;
  input_layer->init_layer(input_size, input_neurons);
  if (chidden_layers > 0) {
    hidden_layers[0] = new Layer;
    hidden_layers[0]->init_layer(input_neurons, hidden_inputs[0]);

    for (unsigned int i = 1; i < chidden_layers; i++) {
      hidden_layers[i] = new Layer;
      hidden_layers[i]->init_layer(hidden_inputs[i - 1], hidden_inputs[i]);
    }
  }
  output_layer = new Layer;
  if (chidden_layers > 0) {
    output_layer->init_layer(hidden_inputs[chidden_layers - 1], output_size);
  } else {
    output_layer->init_layer(input_neurons, output_size);
  }
}
void Network::propagate_network(const float *input) {
  std::memcpy(input_layer->input, input,
              input_layer->count_input * sizeof(float));

  input_layer->propagate_layer();

  if (count_hiddenlayers > 0) {

    for (unsigned int i = 0; i < input_layer->count_neurons; i++) {
      hidden_layers[0]->input[i] = input_layer->neurons[i]->output;
    }

    for (unsigned int i = 0; i < count_hiddenlayers; i++) {

      hidden_layers[i]->propagate_layer();

      for (unsigned int j = 0; j < hidden_layers[i]->count_neurons; j++) {
        if (i < count_hiddenlayers - 1) {
          hidden_layers[i + 1]->input[j] = hidden_layers[i]->neurons[j]->output;
        } else {
          output_layer->input[j] = hidden_layers[i]->neurons[j]->output;
        }
      }
    }
  } else {
    for (unsigned int i = 0; i < input_layer->count_neurons; i++) {
      output_layer->input[i] = input_layer->neurons[i]->output;
    }
  }

  output_layer->propagate_layer();
}

float *Network::getOutput() {
  float *output = new float[output_layer->count_neurons];
  for (unsigned int i = 0; i < output_layer->count_neurons; i++) {
    output[i] = output_layer->neurons[i]->output;
  }
  return output;
}

float Network::train_network(const float *input, const float *awaited_output,
                             const float learning_rate, float momentum) {

  propagate_network(input);

  float sum = 0, csum = 0, errorg = 0;
  float delta, udelta, errorc, output;

  for (unsigned int i = 0; i < output_layer->count_neurons; i++) {
    output = output_layer->neurons[i]->output;
    errorc = (awaited_output[i] - output) * output * (1 - output);
    errorg += (awaited_output[i] - output) * (awaited_output[i] - output);

    for (unsigned int j = 0; j < output_layer->count_input; j++) {

      delta = output_layer->neurons[i]->deltas[j];

      udelta =
          learning_rate * errorc * output_layer->input[j] + delta * momentum;

      output_layer->neurons[i]->weights[j] += udelta;
      output_layer->neurons[i]->deltas[j] =
          learning_rate * errorc * output_layer->input[j];

      sum += output_layer->neurons[i]->weights[j] * errorc;
    }
    output_layer->neurons[i]->wbias +=
        learning_rate * errorc * output_layer->neurons[i]->bias;
  }

  for (int i = (count_hiddenlayers - 1); i >= 0; i--) {
    for (unsigned int j = 0; j < hidden_layers[i]->count_neurons; j++) {
      output = hidden_layers[i]->neurons[j]->output;

      errorc = output * (1 - output) * sum;

      for (unsigned int k = 0; k < hidden_layers[i]->count_input; k++) {
        delta = hidden_layers[i]->neurons[j]->deltas[k];
        udelta = learning_rate * errorc * hidden_layers[i]->input[k] +
                 delta * momentum;
        hidden_layers[i]->neurons[j]->weights[k] += udelta;
        hidden_layers[i]->neurons[j]->deltas[k] =
            learning_rate * errorc * hidden_layers[i]->input[k];
        csum += hidden_layers[i]->neurons[j]->weights[k] * errorc;
      }
    }
    sum = csum;
    csum = 0;
  }

  for (unsigned int i = 0; i < input_layer->count_neurons; i++) {
    output = input_layer->neurons[i]->output;
    errorc = output * (1 - output) * sum;

    for (unsigned int j = 0; j < input_layer->count_input; j++) {
      delta = input_layer->neurons[i]->deltas[j];
      udelta =
          learning_rate * errorc * input_layer->input[j] + delta * momentum;

      input_layer->neurons[i]->weights[j] += udelta;
      input_layer->neurons[i]->deltas[j] =
          learning_rate * errorc * input_layer->input[j];
    }
    input_layer->neurons[i]->wbias +=
        learning_rate * errorc * input_layer->neurons[i]->bias;
  }

  return errorg;
}

Network::~Network() {
  delete input_layer;
  delete output_layer;

  for (unsigned int i = 0; i < count_hiddenlayers; i++) {
    delete hidden_layers[i];
  }
}
