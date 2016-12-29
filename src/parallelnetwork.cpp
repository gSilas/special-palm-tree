#include "parallelnetwork.h"

void ParallelNetwork::init_network(unsigned int *inputs, unsigned int *neurons,
                                   unsigned int clayers) {

  count_layers = clayers;

  layers = new Layer *[clayers];

  for (unsigned int l = 0; l < clayers; l++) {
    layers[l] = new Layer;
    layers[l]->init_layer(inputs[l], neurons[l]);
  }
}

void ParallelNetwork::propagate_network(const float *input) {
  std::memcpy(layers[0]->input, input, layers[0]->count_input * sizeof(float));

  for (unsigned int l = 0; l < count_layers; l++) {

    std::vector<std::thread> threads;

    unsigned int num_threads = std::thread::hardware_concurrency();

    if (num_threads < layers[l]->count_neurons) {
      num_threads = layers[l]->count_neurons;
    }

    int *tiling = new int[num_threads];

    for (unsigned int i = 0; i < layers[l]->count_neurons; i++) {
      tiling[i]++;
    }

    threads.push_back(
        std::thread(&Parallel::tile_propagate_layer, layers[l], 0));
    for (unsigned int i = 1; i < num_threads; i++) {
      threads.push_back(std::thread(&Parallel::tile_propagate_layer, layers[l],
                                    i * tiling[i - 1]));
    }

    for (unsigned int i = 0; i < threads.size(); i++) {
      threads[i].join();
    }

    if (l < count_layers - 1) {
      for (unsigned int n = 0; n < layers[l]->count_neurons; n++) {
        layers[l + 1]->input[n] = layers[l]->neurons[n]->output;
      }
    }
  }
}

float ParallelNetwork::train_network(const float *input,
                                     const float *awaited_output,
                                     const float learning_rate,
                                     float momentum) {

  propagate_network(input);
  float total_error = 0.f;
  float out, delta, dw;

  Layer *output_layer = layers[count_layers - 1];

  for (unsigned int i = 0; i < output_layer->count_neurons; i++) {
    out = output_layer->neurons[i]->output;

    total_error += 0.5f * (awaited_output[i] - out) * (awaited_output[i] - out);

    output_layer->neurons[i]->delta =
        (awaited_output[i] - out) * out * (1 - out);

    output_layer->neurons[i]->wbias +=
        learning_rate * (awaited_output[i] - out) * out * (1 - out);
  }

  for (int l = (int)count_layers - 2; l >= 0; l--) {
    delta = 0.f;
    for (unsigned int i = 0; i < layers[l + 1]->count_neurons; i++) {
      for (unsigned int j = 0; j < layers[l + 1]->count_input; j++) {
        delta += layers[l + 1]->neurons[i]->weights[j] *
                 layers[l + 1]->neurons[i]->delta;
      }
    }
    for (unsigned int n = 0; n < layers[l]->count_neurons; n++) {
      out = layers[l]->neurons[n]->output;

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
      }
    }
  }
  return total_error;
}

ParallelNetwork::~ParallelNetwork() {
  for (unsigned int i = 0; i < count_layers; i++) {
    delete layers[i];
  }
  delete layers;
}
