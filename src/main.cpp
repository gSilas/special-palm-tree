#include <fstream>
#include <iostream>
#include <math.h>
#include <string>

#include "gpunetwork.cuh"
#include "network.h"
#include "parallelnetwork.h"

int ReverseInt(int i) {
  unsigned char ch1, ch2, ch3, ch4;
  ch1 = i & 255;
  ch2 = (i >> 8) & 255;
  ch3 = (i >> 16) & 255;
  ch4 = (i >> 24) & 255;
  return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void writeImage(std::vector<std::vector<float>> &imagevec) {
  std::fstream ifile("test.pgm", std::ios::out);
  if (!ifile.is_open()) {
    std::cerr << "Image::write() : Failed to open!" << std::endl;
  }
  ifile << "P2\n";
  ifile << "# Simple example image\n";
  ifile << 300 << " " << 300 << '\n';
  ifile << 255 << '\n';

  for (int j = 0; j < 20; j++)
    for (int i_row = 0; i_row < 28; ++i_row) {
      for (int i = 0; i < 5; ++i) {
        for (int i_col = 0; i_col < 28; ++i_col) {
          ifile << static_cast<int>(imagevec[i + j * 20][i_col + 28 * i_row] *
                                    255)
                << " ";
        }
        ifile << "0 0 ";
      }
      ifile << '\n';
    }
  if (!ifile.good()) {
    std::cerr << "Image::write() : Failed to write!" << std::endl;
  }
  ifile.close();
}

void read_Mnist(std::string filename, std::vector<std::vector<float>> &vec) {
  std::cout << "READING MNIST IMAGES " << std::endl;
  std::ifstream file(filename, std::ios::binary);
  if (file.is_open()) {
    std::cout << "FILE OPENED" << std::endl;
    int magic_number = 0;
    int number_of_images = 0;
    int n_rows = 0;
    int n_cols = 0;
    file.read((char *)&magic_number, sizeof(magic_number));
    magic_number = ReverseInt(magic_number);
    file.read((char *)&number_of_images, sizeof(number_of_images));
    number_of_images = ReverseInt(number_of_images);
    file.read((char *)&n_rows, sizeof(n_rows));
    n_rows = ReverseInt(n_rows);
    file.read((char *)&n_cols, sizeof(n_cols));
    n_cols = ReverseInt(n_cols);
    for (int i = 0; i < number_of_images; ++i) {
      std::vector<float> tp;
      for (int r = 0; r < n_rows; ++r) {
        for (int c = 0; c < n_cols; ++c) {
          unsigned char temp = 0;
          file.read((char *)&temp, sizeof(temp));
          tp.push_back((float)(temp) / 255);
        }
      }
      vec.push_back(tp);
    }
  }
}

void read_Mnist_Label(std::string filename, std::vector<float> &vec) {
  std::cout << "READING MNIST LABEL " << std::endl;
  std::ifstream file(filename, std::ios::binary);
  if (file.is_open()) {
    std::cout << "FILE OPENEND" << std::endl;
    int magic_number = 0;
    int number_of_images = 0;
    /*int n_rows = 0;*/
    /*int n_cols = 0;*/
    file.read((char *)&magic_number, sizeof(magic_number));
    magic_number = ReverseInt(magic_number);
    file.read((char *)&number_of_images, sizeof(number_of_images));
    number_of_images = ReverseInt(number_of_images);
    for (int i = 0; i < number_of_images; ++i) {
      unsigned char temp = 0;
      file.read((char *)&temp, sizeof(temp));
      vec[i] = (float)temp;
    }
  }
}

void train(ParallelNetwork *net, int epochs, float learning_rate,
           float momentum) {
  std::cout << "TRAINING " << std::endl;
  std::string imagefilename = "mnist/train-images-idx3-ubyte/data";
  std::string labelfilename = "mnist/train-labels-idx1-ubyte/data";
  int number_of_images = 60000;

  float desiredout[60000][10];

  // read MNIST iamge into float vector
  std::vector<std::vector<float>> imagevec;

  read_Mnist(imagefilename, imagevec);

  std::cout << imagevec.size() << std::endl;
  std::cout << imagevec[0].size() << std::endl;

  writeImage(imagevec);

  // read MNIST label into float vector
  std::vector<float> labelvec(number_of_images);
  read_Mnist_Label(labelfilename, labelvec);
  std::cout << labelvec.size() << std::endl;

  for (int i = 0; i < number_of_images; i++) {
    float tmp[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int j = 0; j < 10; j++) {
      if (j < labelvec[i]) {
        tmp[j] = 1.f;
      } else {
        tmp[j] = 0;
      }
    }
    std::memcpy(&desiredout[i][0], tmp, 10 * sizeof(float));
  }
  /*
    for (int i = 0; i < number_of_images; i++) {
      std::cout << "TRAINING "
                << " " << desiredout[i][0] << " " << desiredout[i][1] << " "
                << desiredout[i][2] << " " << desiredout[i][3] << " "
                << desiredout[i][4] << " " << desiredout[i][5] << " "
                << desiredout[i][6] << " " << desiredout[i][7] << " "
                << desiredout[i][8] << " " << desiredout[i][9] << std::endl;
    }*/
  /*  std::cout << " NET RESULT: " << net->layers[2]->neurons[0]->output
            << net->layers[2]->neurons[1]->output
            << net->layers[2]->neurons[2]->output
            << net->layers[2]->neurons[3]->output
            << net->layers[2]->neurons[4]->output
            << net->layers[2]->neurons[5]->output
            << net->layers[2]->neurons[6]->output
            << net->layers[2]->neurons[7]->output
            << net->layers[2]->neurons[8]->output
            << net->layers[2]->neurons[9]->output << std::endl;*/
  std::cout << "TRAINING " << std::endl;
  float error = 0.0;
  for (int i = 0; i < number_of_images; i++) {
    // std::cout << i << std::endl;
    error += (float)net->train_network(&imagevec[i][0], desiredout[i],
                                       learning_rate, momentum);
    /*    std::cout << " NET RESULT: " << net->layers[2]->neurons[0]->output
                  << net->layers[2]->neurons[1]->output
                  << net->layers[2]->neurons[2]->output
                  << net->layers[2]->neurons[3]->output
                  << net->layers[2]->neurons[4]->output
                  << net->layers[2]->neurons[5]->output
                  << net->layers[2]->neurons[6]->output
                  << net->layers[2]->neurons[7]->output
                  << net->layers[2]->neurons[8]->output
                  << net->layers[2]->neurons[9]->output << " " <<
       desiredout[i][0]
                  << desiredout[i][1] << desiredout[i][2] << desiredout[i][3]
                  << desiredout[i][4] << desiredout[i][5] << desiredout[i][6]
                  << desiredout[i][7] << desiredout[i][8] << desiredout[i][9]
                  << std::endl;*/
  }

  error /= number_of_images;

  int j = 0;

  while (error > 0.000001 && j < epochs - 1) {
    std::cout << "EPOCH " << j << std::endl;
    for (int i = 0; i < number_of_images; i++) {
      error += (float)net->train_network(&imagevec[i][0], desiredout[i],
                                         learning_rate, momentum);

      /*std::cout << " NET RESULT: " << net->layers[2]->neurons[0]->output
                << net->layers[2]->neurons[1]->output
                << net->layers[2]->neurons[2]->output
                << net->layers[2]->neurons[3]->output
                << net->layers[2]->neurons[4]->output
                << net->layers[2]->neurons[5]->output
                << net->layers[2]->neurons[6]->output
                << net->layers[2]->neurons[7]->output
                << net->layers[2]->neurons[8]->output
                << net->layers[2]->neurons[9]->output << " " << desiredout[i][0]
                << desiredout[i][1] << desiredout[i][2] << desiredout[i][3]
                << desiredout[i][4] << desiredout[i][5] << desiredout[i][6]
                << desiredout[i][7] << desiredout[i][8] << desiredout[i][9]
                << std::endl;*/
    }
    error /= number_of_images;
    j++;
    std::cout << "ERROR " << error << std::endl;
  }
}

void test(ParallelNetwork *net) {
  std::cout << "TESTING" << std::endl;
  std::string imagefilename = "mnist/t10k-images-idx3-ubyte/data";
  std::string labelfilename = "mnist/t10k-labels-idx1-ubyte/data";
  int number_of_images = 10000;

  // read MNIST iamge into float vector
  std::vector<std::vector<float>> imagevec;
  read_Mnist(imagefilename, imagevec);
  std::cout << imagevec.size() << std::endl;
  std::cout << imagevec[0].size() << std::endl;

  // read MNIST label into float vector
  std::vector<float> labelvec(number_of_images);
  read_Mnist_Label(labelfilename, labelvec);
  std::cout << labelvec.size() << std::endl;

  int success = 0;

  for (int i = 0; i < number_of_images; i++) {
    net->propagate_network(&imagevec[i][0]);

    float out = std::round(
        net->layers[1]->neurons[0].output + net->layers[1]->neurons[1].output +
        net->layers[1]->neurons[2].output + net->layers[1]->neurons[3].output +
        net->layers[1]->neurons[4].output + net->layers[1]->neurons[5].output +
        net->layers[1]->neurons[6].output + net->layers[1]->neurons[7].output +
        net->layers[1]->neurons[8].output + net->layers[1]->neurons[9].output);

    if (out == labelvec[i])
      success++;

    std::cout << "TESTED PATTERN " << i << " DESIRED OUTPUT: " << labelvec[i]
              << " NET RESULT: " << out << std::endl;
  }
  std::cout << "SUCCESS " << success << std::endl;
}

int main(int /*argc*/, char const ** /*argv*/) {
  /*
    float pattern[4][2] = {{0, 0}, {1, 1}, {0, 1}, {1, 0}};
    float desiredout[4][1] = {{0}, {0}, {1}, {1}};
    unsigned int inputs[] = {2, 3};
    unsigned int neurons[] = {3, 1};

    ParallelNetwork *net = new ParallelNetwork;
    net->init_network(inputs, neurons, 2);

    float learning_rate = 0.3f;
    float momentum = 0.8f;

    float error = 0.0;

    for (size_t i = 0; i < 4; i++) {
      error += (float)net->train_network(pattern[i], desiredout[i],
                                          learning_rate, momentum);
    }

    error /= 4;

    int j = 0;

    while (error > 0.01f && j < 20000) {
      std::cout << "EPOCH " << j << std::endl;
      for (size_t i = 0; i < 4; i++) {
        error += (float)net->train_network(pattern[i], desiredout[i],
                                            learning_rate, momentum);
      }
      error /= 4;
      j++;
      std::cout << "ERROR " << error << std::endl;
    }

    for (int i = 0; i < 4; i++) {

      net->propagate_network(pattern[i]);

      std::cout << "TESTED PATTERN " << i << " DESIRED OUTPUT: " <<
    *desiredout[i]
                << " NET RESULT: "
                << std::round(net->layers[1]->neurons[0]->output) << std::endl;
    }
    return 0;*/

  unsigned int inputs[] = {784, 300};
  unsigned int neurons[] = {300, 10};
  float learning_rate = 0.001;
  float momentum = 0.9;

  ParallelNetwork *net = new ParallelNetwork;
  net->init_network(inputs, neurons, 2);

  // 26m49s SUCCESS 9398
  // SUCCESS 9369

  train(net, 5, learning_rate, momentum);
  test(net);
  return 0;
}
