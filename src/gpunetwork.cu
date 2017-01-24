#include "gpunetwork.cuh"

void GPUNetwork::init_network(unsigned int *inputs, unsigned int *neurons,
                              unsigned int clayers) {

  count_layers = clayers;
  num_blocks = new int[clayers];
  threads_per_block = new int[clayers];

  mul_num_blocks = new int[clayers];
  mul_threads_per_block = new int[clayers];

  neuron_size = new int[clayers];
  input_size = new int[clayers];

  std::memcpy(neuron_size, neurons, sizeof(int) * clayers);
  std::memcpy(input_size, inputs, sizeof(int) * clayers);

  for (unsigned int i = 0; i < clayers; i++) {
    std::cout << "Neuron_Size " << i << " " << neuron_size[i] << std::endl;
    std::cout << "Input_Size " << i << " " << input_size[i] << std::endl;
  }

  cudaDeviceProp device_props;
  checkErrorsCuda(cudaGetDeviceProperties(&device_props, 0));
  for (unsigned int l = 0; l < clayers; l++) {
    if (neurons[l] > (unsigned int)device_props.maxThreadsPerBlock)
      threads_per_block[l] = device_props.maxThreadsPerBlock;
    else
      threads_per_block[l] = neurons[l];

    num_blocks[l] = (int)(neurons[l] / threads_per_block[l]);

    if (num_blocks[l] == 0) {
      num_blocks[l]++;
    } else if (0 != neurons[l] % num_blocks[l]) {
      num_blocks[l]++;
    }
    if (neurons[l] * inputs[l] > (unsigned int)device_props.maxThreadsPerBlock)
      mul_threads_per_block[l] = device_props.maxThreadsPerBlock;
    else
      mul_threads_per_block[l] = neurons[l] * inputs[l];

    mul_num_blocks[l] =
        (int)(neurons[l] * inputs[l] / mul_threads_per_block[l]);

    if (mul_num_blocks[l] == 0) {
      mul_num_blocks[l]++;
    } else if (0 != neurons[l] * inputs[l] % mul_num_blocks[l]) {
      mul_num_blocks[l]++;
    }

    // std::cout << "num_blocks = " << num_blocks[l] << std::endl;
  }
  device_inputs = new float *[clayers];

  device_weights = new float *[clayers];
  device_wbias = new float *[clayers];

  device_delta = new float *[clayers];
  device_prvdeltas = new float *[clayers];

  checkErrorsCuda(cudaMalloc((void **)&device_output,
                             sizeof(float) * neurons[count_layers - 1]));

  for (unsigned int i = 0; i < clayers; i++) {
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

    Device::set_dataset<<<1, (set_size / dataset_count)>>>(
        device_inputs[0],
        test_device_dataset + (i * (set_size / dataset_count)), input_size[0]);
    // checkErrorsCuda(cudaDeviceSynchronize());

    Device::tile_propagate_inlayer<<<num_blocks[0], threads_per_block[0]>>>(
        device_inputs[0], device_inputs[1], device_weights[0], device_wbias[0],
        input_size[0], neuron_size[0]);
    // checkErrorsCuda(cudaDeviceSynchronize());

    for (unsigned int l = 1; l < count_layers; l++) {
      if (l >= count_layers - 1) {
        // std::cout << "l115 " << l << std::endl;
        Device::tile_propagate_layer<<<num_blocks[l], threads_per_block[l]>>>(
            device_inputs[l], device_output, device_weights[l], device_wbias[l],
            input_size[l], neuron_size[l]);
      } else {
        // std::cout << "l121 " << l << std::endl;
        Device::tile_propagate_layer<<<num_blocks[l], threads_per_block[l]>>>(
            device_inputs[l], device_inputs[l + 1], device_weights[l],
            device_wbias[l], input_size[l], neuron_size[l]);
      }
      // checkErrorsCuda(cudaDeviceSynchronize());
    }

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
  cudaProfilerStart();
  for (unsigned int e = 0; e < epochs; e++) {

    std::cout << "Epoch " << e + 1 << "/" << epochs << std::endl;

    for (unsigned int i = 0; i < dataset_count; i++) {

      device_awaited_output = device_labels + i * (label_size / dataset_count);

      Device::set_dataset<<<1, (set_size / dataset_count)>>>(
          device_inputs[0], device_dataset + (i * (set_size / dataset_count)),
          input_size[0]);

      // checkErrorsCuda(cudaDeviceSynchronize());

      Device::tile_propagate_inlayer<<<num_blocks[0], threads_per_block[0]>>>(
          device_inputs[0], device_inputs[1], device_weights[0],
          device_wbias[0], input_size[0], neuron_size[0]);
      // checkErrorsCuda(cudaDeviceSynchronize());

      for (unsigned int l = 1; l < count_layers; l++) {
        if (l >= count_layers - 1) {
          // std::cout << "l194 " << l << std::endl;
          Device::tile_propagate_layer<<<num_blocks[l], threads_per_block[l]>>>(
              device_inputs[l], device_output, device_weights[l],
              device_wbias[l], input_size[l], neuron_size[l]);
        } else {
          // std::cout << "l199 " << l << std::endl;
          Device::tile_propagate_layer<<<num_blocks[l], threads_per_block[l]>>>(
              device_inputs[l], device_inputs[l + 1], device_weights[l],
              device_wbias[l], input_size[l], neuron_size[l]);
        }
        // checkErrorsCuda(cudaDeviceSynchronize());
      }

      Device::tile_outlayer_train<<<num_blocks[count_layers - 1],
                                    threads_per_block[count_layers - 1]>>>(
          device_output, device_delta[count_layers - 1],
          device_wbias[count_layers - 1], device_awaited_output, learning_rate,
          input_size[count_layers - 1], neuron_size[count_layers - 1]);
      // checkErrorsCuda(cudaDeviceSynchronize());

      for (int l = (int)count_layers - 2; l > -1; l--) {
        // std::cout << "l215 " << l << std::endl;
        Device::tile_layer_train<<<num_blocks[l], threads_per_block[l]>>>(
            device_inputs[l + 1], device_weights[l + 1], device_delta[l + 1],
            device_wbias[l], device_delta[l], device_awaited_output,
            learning_rate, input_size[l + 1], neuron_size[l + 1], input_size[l],
            neuron_size[l]);
        // checkErrorsCuda(cudaDeviceSynchronize());
      }

      for (unsigned int l = 0; l < count_layers; l++) {
        // std::cout << "l225 " << l << std::endl;
        Device::
            tile_update_layer<<<mul_num_blocks[l], mul_threads_per_block[l]>>>(
                device_inputs[l], device_weights[l], device_delta[l],
                device_prvdeltas[l], learning_rate, momentum, input_size[l],
                neuron_size[l]);
        // checkErrorsCuda(cudaDeviceSynchronize());
      }
    }
  }
  cudaProfilerStop();
}

float *GPUNetwork::getOutput() {

  float *iout = new float[10];
  checkErrorsCuda(cudaMemcpy(iout, device_output, sizeof(float) * 10,
                             cudaMemcpyDeviceToHost));
  for (int i = 0; i < 10; i++) {
    std::cout << "Output on Neuron " << i << " : " << iout[i] << std::endl;
  }
  return iout;
}

GPUNetwork::~GPUNetwork() {
  delete num_blocks;
  delete threads_per_block;
  delete neuron_size;
  delete input_size;

  // device_land
  for (unsigned int i = 0; i < count_layers; i++) {
    checkErrorsCuda(cudaFree(device_inputs[i]));
    checkErrorsCuda(cudaFree(device_weights[i]));
    checkErrorsCuda(cudaFree(device_wbias[i]));
    checkErrorsCuda(cudaFree(device_delta[i]));
    checkErrorsCuda(cudaFree(device_prvdeltas[i]));
  }

  checkErrorsCuda(cudaFree(device_output));
  checkErrorsCuda(cudaFree(device_dataset));
  checkErrorsCuda(cudaFree(test_device_dataset));
  checkErrorsCuda(cudaFree(device_labels));
}
