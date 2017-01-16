#include "gpunetwork.h"

void GPUNetwork::init_network(unsigned int *inputs, unsigned int *neurons,
                              unsigned int clayers) {

  count_layers = clayers;

  layers = new Layer *[clayers];

  for (unsigned int l = 0; l < clayers; l++) {
    layers[l] = new Layer;
    layers[l]->init_layer(inputs[l], neurons[l]);
  }
}

void GPUNetwork::propagate_network(const double *input) {
  std::memcpy(layers[0]->input, input, layers[0]->count_input * sizeof(double));

  for (unsigned int l = 0; l < count_layers; l++) {

    std::vector<std::thread> threads;

    unsigned int num_threads = std::thread::hardware_concurrency();

    int *tiling;
    if (num_threads > layers[l]->count_neurons) {
      num_threads = layers[l]->count_neurons;
      tiling = new int[num_threads];
    } else {
      tiling = new int[num_threads];
    }

    for (unsigned int i = 0; i <= num_threads; i++) {
      tiling[i] = i * std::floor(layers[l]->count_neurons / num_threads);
    }
    /*
        std::cout << "STARTING WITH " << num_threads << std::endl;

        std::cout << tiling[0] << std::endl;
        std::cout << tiling[1] << std::endl;
        std::cout << tiling[2] << std::endl;
        std::cout << tiling[3] << std::endl;
    */
    for (unsigned int i = 0; i < num_threads - 1; i++) {
      /*  std::cout << "THREAD " << i << " with Start " << tiling[i] << " and
         End "
                  << tiling[i + 1] << std::endl;*/
      threads.push_back(std::thread(&Parallel::tile_propagate_layer, layers[l],
                                    tiling[i], tiling[i + 1]));
    }

    threads.push_back(std::thread(&Parallel::tile_propagate_layer, layers[l],
                                  tiling[num_threads - 1],
                                  layers[l]->count_neurons));
    /*
        std::cout << "THREAD " << num_threads - 1 << " with Start "
                  << tiling[num_threads - 1] << " and End "
                  << layers[l]->count_neurons << std::endl;
    */
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

double GPUNetwork::train_network(const double *input,
                                 const double *awaited_output,
                                 const double learning_rate, double momentum) {

  propagate_network(input);

  double total_error = 0;
  double out;

  Layer *output_layer = layers[count_layers - 1];

  for (unsigned int i = 0; i < output_layer->count_neurons; i++) {
    out = output_layer->neurons[i]->output;

    total_error += 0.5 * (awaited_output[i] - out) * (awaited_output[i] - out);

    output_layer->neurons[i]->delta =
        (awaited_output[i] - out) * out * (1 - out);

    output_layer->neurons[i]->wbias +=
        learning_rate * (awaited_output[i] - out) * out * (1 - out);
  }
  // std::cout << "THREAD " << std::endl;

  for (int l = (int)count_layers - 2; l >= 0; l--) {
    std::vector<std::thread> threads;

    unsigned int num_threads = std::thread::hardware_concurrency();

    int *tiling;
    if (num_threads > layers[l]->count_neurons) {
      num_threads = layers[l]->count_neurons;
      tiling = new int[num_threads];
    } else {
      tiling = new int[num_threads];
    }

    for (unsigned int i = 0; i <= num_threads; i++) {
      tiling[i] = i * std::floor(layers[l]->count_neurons / num_threads);
    }
    /*
        std::cout << "STARTING WITH " << num_threads << std::endl;

        std::cout << tiling[0] << std::endl;
        std::cout << tiling[1] << std::endl;
        std::cout << tiling[2] << std::endl;
        std::cout << tiling[3] << std::endl;
    */
    for (unsigned int i = 0; i < num_threads - 1; i++) {
      threads.push_back(std::thread(&Parallel::tile_layer_train, layers[l],
                                    layers[l + 1], tiling[i], tiling[i + 1],
                                    learning_rate));
    }

    threads.push_back(std::thread(&Parallel::tile_layer_train, layers[l],
                                  layers[l + 1], tiling[num_threads - 1],
                                  layers[l]->count_neurons, learning_rate));

    for (unsigned int i = 0; i < threads.size(); i++) {
      threads[i].join();
    }
  }

  for (unsigned int l = 0; l < count_layers; l++) {
    std::vector<std::thread> threads;

    unsigned int num_threads = std::thread::hardware_concurrency();

    int *tiling;
    if (num_threads > layers[l]->count_neurons) {
      num_threads = layers[l]->count_neurons;
      tiling = new int[num_threads];
    } else {
      tiling = new int[num_threads];
    }

    for (unsigned int i = 0; i <= num_threads; i++) {
      tiling[i] = i * std::floor(layers[l]->count_neurons / num_threads);
    }
    /*
        std::cout << "STARTING WITH " << num_threads << std::endl;

        std::cout << tiling[0] << std::endl;
        std::cout << tiling[1] << std::endl;
        std::cout << tiling[2] << std::endl;
        std::cout << tiling[3] << std::endl;
    */
    for (unsigned int i = 0; i < num_threads; i++) {
      threads.push_back(std::thread(&Parallel::tile_layer_update, layers[l],
                                    tiling[i], tiling[i + 1], learning_rate,
                                    momentum));
    }

    threads.push_back(std::thread(
        &Parallel::tile_layer_update, layers[l], tiling[num_threads - 1],
        layers[l]->count_neurons, learning_rate, momentum));

    for (unsigned int i = 0; i < threads.size(); i++) {
      threads[i].join();
    }
  }
  return total_error;
}

GPUNetwork::~GPUNetwork() {
  for (unsigned int i = 0; i < count_layers; i++) {
    delete layers[i];
  }
  delete layers;
}
