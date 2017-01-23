#include "gpunetwork.cuh"

void GPUNetwork::init_network(unsigned int *inputs, unsigned int *neurons,
                              unsigned int clayers) {

  count_layers = clayers;
  num_blocks = new int[clayers];
  threads_per_block = new int[clayers];

  cudaDeviceProp device_props;
  checkErrorsCuda(cudaGetDeviceProperties(&device_props, 0));
  for (unsigned int l = 0; l < clayers; l++) {
    if (neurons[l] > (unsigned int)device_props.maxThreadsPerBlock)
      threads_per_block[l] = device_props.maxThreadsPerBlock;
    else
      threads_per_block[l] = neurons[l];

    num_blocks[l] = (int)(neurons[l] / threads_per_block[l]);

    if (num_blocks[l] == 0)
      num_blocks[l]++;
    else if (0 != neurons[l] % num_blocks[l])
      num_blocks[l]++;

    //std::cout << "num_blocks = " << num_blocks[l] << std::endl;
  }
  device_inputs = new float *[clayers];

  device_weights = new float *[clayers];
  device_wbias = new float *[clayers];

  device_delta = new float *[clayers];
  device_prvdeltas = new float *[clayers];

  checkErrorsCuda(cudaMalloc((void **)&device_output,
                             sizeof(float) * neurons[count_layers - 1]));

  for (int i = 0; i < clayers; i++) {
    checkErrorsCuda(
        cudaMalloc((void **)&device_inputs[i], sizeof(float) * inputs[i]));
    checkErrorsCuda(
        cudaMalloc((void **)&device_wbias[i], sizeof(float) * neurons[i]));
    checkErrorsCuda(
        cudaMalloc((void **)&device_delta[i], sizeof(float) * neurons[i]));

    checkErrorsCuda(cudaMalloc((void **)&device_weights[i],
                               sizeof(float) * inputs[i] * neurons[i]));
    checkErrorsCuda(cudaMalloc((void **)&device_prvdeltas[i],
                               sizeof(float) * inputs[i] * neurons[i]));

    Device::set_layer_memory<<<num_blocks[i], threads_per_block[i]>>>(
        device_delta[i], device_prvdeltas[i], inputs[i], neurons[i]);

    float *weights = new float[inputs[i] * neurons[i]];
    float *wbias = new float[neurons[i]];

    for (unsigned int j = 0; j < inputs[i] * neurons[i]; j++) {
      weights[j] = -0.5 +
                   static_cast<float>(rand()) /
                       (static_cast<float>(RAND_MAX / (0.5 - (-0.5))));
    }
    for (unsigned int j = 0; j < neurons[i]; j++) {
      wbias[j] = -0.5 +
                 static_cast<float>(rand()) /
                     (static_cast<float>(RAND_MAX / (0.5 - (-0.5))));
    }

    checkErrorsCuda(cudaMemcpy(device_weights[i], weights,
                               sizeof(float) * inputs[i] * neurons[i],
                               cudaMemcpyHostToDevice));
    checkErrorsCuda(cudaMemcpy(device_wbias[i], wbias,
                               sizeof(float) * neurons[i],
                               cudaMemcpyHostToDevice));
    delete weights;
    delete wbias;
  }
}

unsigned int GPUNetwork::propagate_network(float *data_set, float *label_set,
                                           unsigned int dataset_count,
                                           size_t set_size, size_t label_size) {
  unsigned int success = 0;

  checkErrorsCuda(
      cudaMalloc((void **)&test_device_dataset, sizeof(float) * set_size));

  checkErrorsCuda(cudaMemcpy(test_device_dataset, data_set,
                             sizeof(float) * set_size, cudaMemcpyHostToDevice));

  for (unsigned int i = 0; i < dataset_count; i++) {

    Device::tile_propagate_inlayer<<<num_blocks[0], threads_per_block[0]>>>(
        test_device_dataset + (i * (set_size / dataset_count)),
        device_inputs[0], device_inputs[1], device_weights[0], device_wbias[0],
        784, 300);
    checkErrorsCuda(cudaDeviceSynchronize());
    Device::tile_propagate_layer<<<num_blocks[1], threads_per_block[1]>>>(
        device_inputs[1], device_output, device_weights[1], device_wbias[1],
        300, 10);
    checkErrorsCuda(cudaDeviceSynchronize());

    float *out;
    out = getOutput();

    float outf = -1;
    float index = 0;

    for (int j = 0; j < 10; j++) {
      if (out[j] > outf) {
        outf = out[j];
        index = j;
      }
    }

    float desired = 0;
    for (int j = 0; j < 10; j++) {
      if (label_set[j + i * (label_size / dataset_count)] == 1) {
        desired = j;
      }
    }

    if (index == desired)
      success++;
    std::cout << "Pattern:  " << i << "  Label:  " << desired
              << " Neural Net Result:  " << index
              << "  Neural Net Output:  " << outf << std::endl;
  }
  return success;
}

void GPUNetwork::train_network(float *data_set, size_t set_size,
                               float *data_labels, size_t label_size,
                               unsigned int dataset_count, unsigned int epochs,
                               const float learning_rate, float momentum) {

  checkErrorsCuda(
      cudaMalloc((void **)&device_dataset, sizeof(float) * set_size));
  checkErrorsCuda(
      cudaMalloc((void **)&device_labels, sizeof(float) * label_size));

  checkErrorsCuda(cudaMemcpy(device_dataset, data_set, sizeof(float) * set_size,
                             cudaMemcpyHostToDevice));
  checkErrorsCuda(cudaMemcpy(device_labels, data_labels,
                             sizeof(float) * label_size,
                             cudaMemcpyHostToDevice));
  for (unsigned int e = 0; e < epochs; e++) {

    std::cout << "Epoch " << e + 1 << "/" << epochs << std::endl;
    for (unsigned int i = 0; i < dataset_count; i++) {

      device_awaited_output = device_labels + i * (label_size / dataset_count);

      Device::tile_propagate_inlayer<<<num_blocks[0], threads_per_block[0]>>>(
          device_dataset + (i * (set_size / dataset_count)), device_inputs[0],
          device_inputs[1], device_weights[0], device_wbias[0], 784, 300);
      checkErrorsCuda(cudaDeviceSynchronize());
      Device::tile_propagate_layer<<<num_blocks[1], threads_per_block[1]>>>(
          device_inputs[1], device_output, device_weights[1], device_wbias[1],
          300, 10);
      checkErrorsCuda(cudaDeviceSynchronize());

      Device::tile_outlayer_train<<<num_blocks[1], threads_per_block[1]>>>(
          device_output, device_delta[1], device_wbias[1],
          device_awaited_output, learning_rate, 300, 10);
      checkErrorsCuda(cudaDeviceSynchronize());
      Device::tile_layer_train<<<num_blocks[0], threads_per_block[0]>>>(
          device_inputs[1], device_weights[1], device_delta[1], device_wbias[0],
          device_delta[0], device_awaited_output, learning_rate, 300, 10, 784,
          300);
      checkErrorsCuda(cudaDeviceSynchronize());

      Device::tile_update_layer<<<num_blocks[0], threads_per_block[0]>>>(
          device_inputs[0], device_weights[0], device_delta[0],
          device_prvdeltas[0], learning_rate, momentum, 784, 300);
      checkErrorsCuda(cudaDeviceSynchronize());
      Device::tile_update_layer<<<num_blocks[1], threads_per_block[1]>>>(
          device_inputs[1], device_weights[1], device_delta[1],
          device_prvdeltas[1], learning_rate, momentum, 300, 10);
      checkErrorsCuda(cudaDeviceSynchronize());
    }
  }
}

float *GPUNetwork::getOutput() {
  // float *out = new float[sum_neuron_size[count_layers]];
  /*float *iout = new float[sum_input_size[count_layers] +
                          arr_neuron_size[count_layers - 1]];
  float *wout = new float[sum_weight_size[count_layers]];
  float *dout = new float[sum_neuron_size[count_layers]];

  checkErrorsCuda(cudaMemcpy(dout, device_delta,
                             sizeof(float) * sum_neuron_size[count_layers],
                             cudaMemcpyDeviceToHost));
  checkErrorsCuda(cudaMemcpy(iout, device_input,
                             sizeof(float) * sum_input_size[count_layers] +
                                 arr_neuron_size[count_layers - 1],
                             cudaMemcpyDeviceToHost));
  checkErrorsCuda(cudaMemcpy(wout, device_weights,
                             sizeof(float) * sum_weight_size[count_layers],
                             cudaMemcpyDeviceToHost));
  for (int i = 0; i < sum_neuron_size[count_layers]; i++) {
    std::cout << "d " << i << " " << dout[i] << std::endl;
  }

  for (int i = 0;
       i < sum_input_size[count_layers] + arr_neuron_size[count_layers - 1];
       i++) {
    std::cout << "i " << i << " " << iout[i] << std::endl;
  }
  for (int i = 0; i < sum_weight_size[count_layers]; i++) {
    std::cout << "w " << i << " " << wout[i] << std::endl;
  }
  delete iout;
  delete wout;
  delete dout;*/
  float *iout = new float[10];
  checkErrorsCuda(cudaMemcpy(iout, device_output, sizeof(float) * 10,
                             cudaMemcpyDeviceToHost));
  for (int i = 0; i < 10; i++) {
    std::cout << "Output on Neuron " << i << " : " << iout[i] << std::endl;
  }
  return iout;
}

GPUNetwork::~GPUNetwork() {
  /*  delete num_blocks;cuMemsetD32
    delete threads_per_block;
    delete arr_input_size;
    delete arr_neuron_size;
    delete sum_input_size;
    delete sum_neuron_size;
    delete sum_weight_size;

    // device_land
    checkErrorsCuda(cudaFree(device_input));

    checkErrorsCuda(cudaFree(device_weights));
    checkErrorsCuda(cudaFree(device_wbias));

    checkErrorsCuda(cudaFree(device_delta));
    checkErrorsCuda(cudaFree(device_prvdeltas));

    checkErrorsCuda(cudaFree(device_dataset));
    checkErrorsCuda(cudaFree(device_labels));

    checkErrorsCuda(cudaFree(test_device_dataset)); */
}
