
#include <iostream>

#include "network.h"
#include "parallelnetwork.h"

int main(int /*argc*/, char const ** /*argv*/) {

  float pattern[4][2] = {{0, 0}, {1, 1}, {0, 1}, {1, 0}};
  float desiredout[4][1] = {{0}, {0}, {1}, {1}};
  unsigned int inputs[] = {2, 3};
  unsigned int neurons[] = {3, 1};

  Network *net = new Network;
  net->init_network(inputs, neurons, 2);

  float learning_rate = 0.3f;
  float momentum = 0.8f;

  long double error = 0.0;

  for (size_t i = 0; i < 4; i++) {
    error += (long double)net->train_network(pattern[i], desiredout[i],
                                             learning_rate, momentum);
  }

  error /= 4;

  int j = 0;

  while (error > 0.001f && j < 20000) {
    std::cout << "Epoch " << j << std::endl;
    for (size_t i = 0; i < 4; i++) {
      error += (long double)net->train_network(pattern[i], desiredout[i],
                                               learning_rate, momentum);
    }
    error /= 4;
    j++;
    std::cout << "Error " << error << std::endl;
  }

  for (int i = 0; i < 4; i++) {

    net->propagate_network(pattern[i]);
    /*
        for (unsigned int l = 0; l < net->count_layers; l++) {
          std::cout << std::endl
                    << "-----" << std::endl
                    << "Layer: " << l << std::endl
                    << "NeuronCount " << net->layers[l]->count_neurons <<
       std::endl;
          for (unsigned int n = 0; n < net->layers[l]->count_neurons; n++) {
            std::cout << std::endl
                      << "Neuron: " << n << std::endl
                      << "Output " << net->layers[l]->neurons[n]->output
                      << std::endl;
            for (unsigned int i = 0; i < net->layers[l]->count_input; i++) {
              std::cout << std::endl
                        << "Input: " << i << std::endl
                        << "Input " << net->layers[l]->input[i] << std::endl
                        << "Weight " << net->layers[l]->neurons[n]->weights[i]
                        << std::endl;
            }
          }
        }*/

    std::cout << "Input " << i << " Expected Output: " << *desiredout[i]
              << " Network Output: " << net->layers[1]->neurons[0]->output
              << " Rounded Network Output: "
              << std::round(net->layers[1]->neurons[0]->output) << std::endl;
  }
  return 0;
}
