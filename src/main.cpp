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

void read_Mnist(std::string filename, std::vector<float> &vec) {
  std::cout << "Reading MNIST pixels!" << std::endl;
  std::ifstream file(filename, std::ios::binary);
  if (file.is_open()) {
    std::cout << "MNIST pixel file opened!" << std::endl;
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
      for (int r = 0; r < n_rows; ++r) {
        for (int c = 0; c < n_cols; ++c) {
          unsigned char temp = 0;
          file.read((char *)&temp, sizeof(temp));
          vec.push_back((float)(temp) / 255);
        }
      }
    }
  } else {
    std::cout << "MNIST pixel file not opened!" << std::endl;
  }
}

void read_Mnist_Label(std::string filename, std::vector<float> &vec) {
  std::cout << "Reading MNIST labels!" << std::endl;
  std::ifstream file(filename, std::ios::binary);
  if (file.is_open()) {
    std::cout << "MNIST label file opened!" << std::endl;
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
      // std::cout << (int)temp << std::endl;
      for (int j = 0; j < 10; j++) {
        if ((int)temp != j) {
          vec.push_back(0.f);
          // std::cout << "0 ";
        } else {
          vec.push_back(1.f);
          // std::cout << "1 ";
        }
      }
      // std::cout << std::endl;
    }
  } else {
    std::cout << "MNIST label file not opened!" << std::endl;
  }
}

void train(GPUNetwork *net, int epochs, float learning_rate, float momentum) {
  std::cout << "Training function called!" << std::endl;
  std::string imagefilename = "mnist/train-images-idx3-ubyte/data";
  std::string labelfilename = "mnist/train-labels-idx1-ubyte/data";
  int number_of_images = 60000;

  // read MNIST iamge into float vector
  std::vector<float> imagevec;

  read_Mnist(imagefilename, imagevec);

  std::cout << "Size of image file: " << imagevec.size() << std::endl;

  // read MNIST label into float vector
  std::vector<float> labelvec;

  read_Mnist_Label(labelfilename, labelvec);
  std::cout << "Size of label file: " << labelvec.size() << std::endl;

  net->train_network(&imagevec[0], (size_t)imagevec.size(), &labelvec[0],
                     (size_t)labelvec.size(), number_of_images, epochs,
                     learning_rate, momentum);
}

void test(GPUNetwork *net) {
  std::cout << "Testing function called!" << std::endl;
  std::string imagefilename = "mnist/t10k-images-idx3-ubyte/data";
  std::string labelfilename = "mnist/t10k-labels-idx1-ubyte/data";
  int number_of_images = 10000;

  // read MNIST iamge into float vector
  std::vector<float> imagevec;
  read_Mnist(imagefilename, imagevec);
  std::cout << "Size of image file: " << imagevec.size() << std::endl;

  // read MNIST label into float vector
  std::vector<float> labelvec;
  read_Mnist_Label(labelfilename, labelvec);
  std::cout << "Size of label file: " << labelvec.size() << std::endl;

  int success =
      net->propagate_network(&imagevec[0], &labelvec[0], number_of_images,
                             (size_t)imagevec.size(), (size_t)labelvec.size());

  std::cout << "Successfully Recognized:  " << success << "  Images"
            << std::endl;
  std::cout << "Accuracy:  " << success / number_of_images << std::endl;
}

int main(int /*argc*/, char const ** /*argv*/) {

  unsigned int inputs[] = {784, 300};
  unsigned int neurons[] = {300, 10};

  GPUNetwork *net = new GPUNetwork;
  net->init_network(inputs, neurons, 2);

  float learning_rate = 0.001;
  float momentum = 0.8;

  // CPU 26m49s SUCCESS 9398
  // SUCCESS 9369

  train(net, 6, learning_rate, momentum);
  test(net);

  delete net;
  return 0;
}
