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
  std::cout << "Reading MNIST images!" << std::endl;
  std::ifstream file(filename, std::ios::binary);
  if (file.is_open()) {
    std::cout << "File opened!" << std::endl;
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
  std::cout << "Reading MNIST labels!" << std::endl;
  std::ifstream file(filename, std::ios::binary);
  if (file.is_open()) {
    std::cout << "File opened!" << std::endl;
    int magic_number = 0;
    int number_of_images = 0;
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
  std::cout << "Training started!" << std::endl;
  std::string imagefilename = "mnist/train-images-idx3-ubyte/data";
  std::string labelfilename = "mnist/train-labels-idx1-ubyte/data";
  int number_of_images = 60000;

  float desiredout[60000][10];

  std::vector<std::vector<float>> imagevec;

  read_Mnist(imagefilename, imagevec);

  std::cout << imagevec.size() << std::endl;
  std::cout << imagevec[0].size() << std::endl;

  writeImage(imagevec);

  std::vector<float> labelvec(number_of_images);
  read_Mnist_Label(labelfilename, labelvec);
  std::cout << labelvec.size() << std::endl;

  for (int i = 0; i < number_of_images; i++) {
    float tmp[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::cout << labelvec[i] << std::endl;
    for (int j = 0; j < 10; j++) {
      if (j < labelvec[i]) {
        tmp[j] = 1.f;
        std::cout << "1";

      } else {
        tmp[j] = 0;
        std::cout << "0";
      }
    }
    std::cout << std::endl;
    std::memcpy(&desiredout[i][0], tmp, 10 * sizeof(float));
  }
  int j = 1;
  float error = 0.0;
  std::cout << "Epoch: " << j << "/" << epochs << std::endl;
  for (int i = 0; i < number_of_images; i++) {
    error += (float)net->train_network(&imagevec[i][0], desiredout[i],
                                       learning_rate, momentum);
  }
  j++;
  error /= number_of_images;

  while (error > 0.000001 && j < epochs + 1) {
    std::cout << "Epoch: " << j << "/" << epochs << std::endl;
    for (int i = 0; i < number_of_images; i++) {
      error += (float)net->train_network(&imagevec[i][0], desiredout[i],
                                         learning_rate, momentum);
    }
    error /= number_of_images;
    j++;
    std::cout << "Error: " << error << std::endl;
  }
}

void test(ParallelNetwork *net) {
  std::cout << "Testing started!" << std::endl;
  std::string imagefilename = "mnist/t10k-images-idx3-ubyte/data";
  std::string labelfilename = "mnist/t10k-labels-idx1-ubyte/data";
  int number_of_images = 10000;

  std::vector<std::vector<float>> imagevec;
  read_Mnist(imagefilename, imagevec);
  std::cout << imagevec.size() << std::endl;
  std::cout << imagevec[0].size() << std::endl;

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

    std::cout << "Tested Image: " << i << " Desired label: " << labelvec[i]
              << " Network Result: " << out << std::endl;
  }
  std::cout << "Successfull Recognitions: " << success << std::endl;
}

int main(int /*argc*/, char const ** /*argv*/) {

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
