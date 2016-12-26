
#include <iostream>

#include "network.h"

int main(int /*argc*/, char const ** /*argv*/) {
  /*
    if (argc <= 1) {
      std::cerr << "No valid input! Use 'h' for help!" << std::endl;
      return EXIT_FAILURE;
    }*/

  float pattern[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  float desiredout[4][1] = {{0}, {1}, {1}, {0}};
  unsigned int inputs[] = {2, 3};
  unsigned int neurons[] = {3, 1};

  Network *net = new Network;
  net->init_network(inputs, neurons, 0);

  float learning_rate = 0.9f;
  float momentum = 0.f;

  double error = 0.0;
  int i = 0;

  error += (double)net->train_network(pattern[0], desiredout[0], learning_rate,
                                      momentum);
  error += (double)net->train_network(pattern[1], desiredout[1], learning_rate,
                                      momentum);
  error += (double)net->train_network(pattern[2], desiredout[2], learning_rate,
                                      momentum);
  error += (double)net->train_network(pattern[3], desiredout[3], learning_rate,
                                      momentum);
  error /= 4;

  while (error > 0.0001f && i < 50000) {
    std::cout << "EPOCH " << i << std::endl;
    // std::cout << "LEARNIGN RATE " << learning_rate << std::endl;
    // std::cout << "MOMENTUM " << momentum << std::endl;

    error += (double)net->train_network(pattern[0], desiredout[0],
                                        learning_rate, momentum);
    error += (double)net->train_network(pattern[1], desiredout[1],
                                        learning_rate, momentum);
    error += (double)net->train_network(pattern[2], desiredout[2],
                                        learning_rate, momentum);
    error += (double)net->train_network(pattern[3], desiredout[3],
                                        learning_rate, momentum);
    error /= 4;
    i++;
    std::cout << "ERROR " << error << std::endl;
  }

  for (int i = 0; i < 4; i++) {

    net->propagate_network(pattern[i]);

    std::cout << "TESTED PATTERN " << i << " DESIRED OUTPUT: " << *desiredout[i]
              << " NET RESULT: " << net->getOutput()[0] << std::endl;
  }

  return 0;
}
