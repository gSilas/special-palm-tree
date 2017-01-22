#include "gpunetwork.cuh"

void GPUNetwork::init_network(unsigned int *inputs, unsigned int *neurons,
                              unsigned int clayers) {

  count_layers = clayers;
  arr_input_size = new unsigned int[clayers];
  arr_neuron_size = new unsigned int[clayers];
  sum_input_size = new unsigned int[clayers + 1];
  sum_neuron_size = new unsigned int[clayers + 1];
  sum_weight_size = new unsigned int[clayers + 1];
  sum_input_size[0] = 0;
  sum_neuron_size[0] = 0;
  sum_weight_size[0] = 0;
  num_blocks = new int[clayers];
  threads_per_block = new int[clayers];

  std::memcpy(arr_input_size, inputs, sizeof(unsigned int) * clayers);
  std::memcpy(arr_neuron_size, neurons, sizeof(unsigned int) * clayers);

  cudaDeviceProp device_props;
  checkErrorsCuda(cudaGetDeviceProperties(&device_props, 0));

  size_t input_size = 0;
  size_t neuron_size = 0;
  size_t weight_size = 0;

  for (unsigned int l = 0; l < clayers; l++) {
    input_size += inputs[l];
    neuron_size += neurons[l];

    weight_size += inputs[l] * neurons[l];
    if (neurons[l] > (unsigned int)device_props.maxThreadsPerBlock)
      threads_per_block[l] = device_props.maxThreadsPerBlock;
    else
      threads_per_block[l] = neurons[l];

    num_blocks[l] = (int)(neurons[l] / threads_per_block[l]);

    if (num_blocks[l] == 0)
      num_blocks[l]++;
    else if (0 != neurons[l] % num_blocks[l])
      num_blocks[l]++;

    std::cout << "num_blocks = " << num_blocks[l] << std::endl;
  }

  for (unsigned int l = 1; l < clayers + 1; l++) {
    sum_input_size[l] = inputs[l - 1] + sum_input_size[l - 1];
    sum_neuron_size[l] = neurons[l - 1] + sum_neuron_size[l - 1];
    sum_weight_size[l] = neurons[l - 1] * inputs[l - 1];
  }

  for (unsigned int l = 0; l < clayers + 1; l++) {
    std::cout << l << " | " << sum_input_size[l] << " | " << sum_neuron_size[l]
              << std::endl;
  }

  std::cout << input_size << " | " << neuron_size << " | " << weight_size
            << std::endl
            << "cudaMalloc" << std::endl;

  checkErrorsCuda(cudaMalloc(
      (void **)&device_input,
      sizeof(float) * (input_size + arr_neuron_size[count_layers - 1])));
  checkErrorsCuda(
      cudaMalloc((void **)&device_wbias, sizeof(float) * neuron_size));
  checkErrorsCuda(
      cudaMalloc((void **)&device_delta, sizeof(float) * neuron_size));

  checkErrorsCuda(
      cudaMalloc((void **)&device_weights, sizeof(float) * weight_size));
  checkErrorsCuda(
      cudaMalloc((void **)&device_prvdeltas, sizeof(float) * weight_size));

  std::cout << "cudaMemset" << std::endl;

  checkErrorsCuda(cudaMemset(device_input, 0, sizeof(float) * (input_size + arr_neuron_size[count_layers - 1])));

  checkErrorsCuda(cudaMemset(device_wbias, 1, sizeof(float) * neuron_size));
  checkErrorsCuda(cudaMemset(device_delta, 0, sizeof(float) * neuron_size));

  checkErrorsCuda(cudaMemset(device_weights, 0, sizeof(float) * weight_size));
  checkErrorsCuda(cudaMemset(device_prvdeltas, 0, sizeof(float) * weight_size));
  /*
    float *iout = new float[sum_input_size[count_layers]];
    float *wout = new float[sum_weight_size[count_layers]];

    checkErrorsCuda(cudaMemcpy(iout, device_input,
                               sizeof(float) * sum_input_size[count_layers],
                               cudaMemcpyDeviceToHost));
    checkErrorsCuda(cudaMemcpy(wout, device_weights,
                               sizeof(float) * sum_weight_size[count_layers],
                               cudaMemcpyDeviceToHost));

    for (int i = 0; i < sum_input_size[count_layers]; i++) {
      std::cout <<"i "<< iout[i] << std::endl;
    }
    for (int i = 0; i < sum_weight_size[count_layers]; i++) {
      std::cout<<"w " << wout[i] << std::endl;
    }*/
}
unsigned int GPUNetwork::propagate_network(float *data_set, float *label_set,
                                           unsigned int dataset_count,
                                           size_t set_size, size_t label_size) {
  unsigned int success = 0;

  checkErrorsCuda(
      cudaMalloc((void **)&test_device_dataset, sizeof(float) * set_size));

  checkErrorsCuda(cudaMemcpy(test_device_dataset, data_set,
                             sizeof(float) * set_size, cudaMemcpyHostToDevice));

  for (unsigned int i = 10; i < dataset_count; i++) {
    Device::tile_propagate_inlayer<<<num_blocks[0], threads_per_block[0]>>>(
        test_device_dataset + (i * (set_size / dataset_count)), device_input,
        device_weights, device_wbias, arr_input_size[0], arr_neuron_size[0]);

    // checkErrorsCuda(cudaDeviceSynchronize());

    for (int l = 2; l < count_layers + 1; l++) {

      Device::
          tile_propagate_layer<<<num_blocks[l - 1], threads_per_block[l - 1]>>>(
              device_input, device_weights, device_wbias, arr_input_size[l - 1],
              arr_neuron_size[l - 1], sum_input_size[l - 1],
              sum_neuron_size[l - 1], sum_neuron_size[l],
              sum_weight_size[l - 1]);

      /*  std::cout << l << " | " << num_blocks[l - 1] << " | "
                  << threads_per_block[l - 1] << " | " << arr_input_size[l - 1]
                  << " | " << arr_neuron_size[l - 1] << " | "
                  << sum_input_size[l - 1] << " | " << sum_neuron_size[l - 1]
                  << " | " << sum_neuron_size[l] << " | "
                  << sum_weight_size[l - 1] << std::endl;*/

      // checkErrorsCuda(cudaDeviceSynchronize());
    }
    /*  float *out;
      out = getOutput();

      float outf = 0;

      for (int j = 0; j < 10; j++) {
        outf += out[j + sum_neuron_size[count_layers - 1]];
      }
      float desired = 0;
      for (int j = 0; j < 10; j++) {
        desired += label_set[j + i * (label_size / dataset_count)];
      }

      if (std::round(outf) == desired)
        success++;
      std::cout << "TESTED PATTERN " << i << " DESIRED OUTPUT: " << desired
                << " NET RESULT: " << outf << std::endl;
    }*/
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
    std::cout << "EPOCH " << e << std::endl;
    for (unsigned int i = 0; i < dataset_count; i++) {

      device_awaited_output = device_labels + i * (label_size / dataset_count);

      Device::tile_propagate_inlayer<<<num_blocks[0], threads_per_block[0]>>>(
          device_dataset + (i * (set_size / dataset_count)), device_input,
          device_weights, device_wbias, arr_input_size[0], arr_neuron_size[0]);

      // checkErrorsCuda(cudaDeviceSynchronize());

      for (int l = 1; l < count_layers - 1; l++) {

        Device::tile_propagate_layer<<<num_blocks[l - 1],
                                       threads_per_block[l - 1]>>>(
            device_input, device_weights, device_wbias, arr_input_size[l - 1],
            arr_neuron_size[l - 1], sum_input_size[l - 1],
            sum_neuron_size[l - 1], sum_neuron_size[l]+arr_input_size[0], sum_weight_size[l - 1]);

        // checkErrorsCuda(cudaDeviceSynchronize());
        /*std::cout << l << " | " << num_blocks[l] << " | " <<
           threads_per_block[l]
                  << " | " << arr_input_size[l] << " | " << arr_neuron_size[l]
                  << " | " << sum_input_size[l] << " | " << sum_neuron_size[l]
                  << " | " << sum_weight_size[l] << " | " << std::endl;*/
      }

      Device::tile_outlayer_train<<<num_blocks[count_layers - 1],
                                    threads_per_block[count_layers - 1]>>>(
          device_input, device_delta, device_wbias, device_awaited_output,
          learning_rate, sum_neuron_size[count_layers - 1] + arr_input_size[0]);

      // checkErrorsCuda(cudaDeviceSynchronize());

      for (int l = count_layers - 2; l >= 0; l--) {
        Device::tile_layer_train<<<num_blocks[l], threads_per_block[l]>>>( device_input,
            device_weights, device_wbias, device_delta, device_awaited_output,
            learning_rate, arr_neuron_size[l + 1], arr_input_size[l + 1],
            sum_weight_size[l + 1], sum_neuron_size[l + 1], arr_input_size[0],
            sum_neuron_size[l]);

        // checkErrorsCuda(cudaDeviceSynchronize());
        /*
            std::cout << " tile_layer_train " << std::endl;
            std::cout << l << " | " << num_blocks[l] << " | " <<
           threads_per_block[l]
                      << " | " << arr_neuron_size[l + 1] << " | "
                      << arr_input_size[l + 1] << " | " << sum_weight_size[l +
           1]
                      << " | " << sum_neuron_size[l + 1] << " | " <<
           sum_neuron_size[l]
                      << std::endl; */
      }

      for (int l = 0; l < count_layers; l++) {
        Device::tile_update_layer<<<num_blocks[l], threads_per_block[l]>>>(
            device_input, device_weights, device_delta, device_prvdeltas,
            learning_rate, momentum, sum_input_size[l], sum_neuron_size[l],
            arr_input_size[l], sum_weight_size[l]);

        // checkErrorsCuda(cudaDeviceSynchronize());
        /*
            std::cout << " tile_update_layer " << std::endl;
            std::cout << l << " | " << num_blocks[l] << " | " <<
           threads_per_block[l]
                      << " | " << sum_input_size[l] << " | " <<
           sum_neuron_size[l]
                      << " | " << arr_input_size[l] << " | " <<
           sum_weight_size[l]
                      << std::endl; */
      }
    }
    getOutput();
  }
}

float *GPUNetwork::getOutput() {
  float *out = new float[sum_neuron_size[count_layers]];
  float *iout = new float[sum_input_size[count_layers]+arr_neuron_size[count_layers-1]];
  float *wout = new float[sum_weight_size[count_layers]];
  float *dout = new float[sum_neuron_size[count_layers]];

  checkErrorsCuda(cudaMemcpy(dout, device_delta,
                             sizeof(float) * sum_neuron_size[count_layers],
                             cudaMemcpyDeviceToHost));
  checkErrorsCuda(cudaMemcpy(iout, device_input,
                             sizeof(float) * sum_input_size[count_layers]+arr_neuron_size[count_layers-1],
                             cudaMemcpyDeviceToHost));
  checkErrorsCuda(cudaMemcpy(wout, device_weights,
                             sizeof(float) * sum_weight_size[count_layers],
                             cudaMemcpyDeviceToHost));
  for (int i = 0; i < sum_neuron_size[count_layers]; i++) {
    std::cout << "d " << i << " " << dout[i] << std::endl;
  }

  for (int i = 0; i < sum_input_size[count_layers]+arr_neuron_size[count_layers-1]; i++) {
    std::cout << "i " << i << " " << iout[i] << std::endl;
  }
  for (int i = 0; i < sum_weight_size[count_layers]; i++) {
    std::cout << "w " << i << " " << wout[i] << std::endl;
  }
  delete iout;
  delete wout;
  delete dout;
  checkErrorsCuda(cudaMemcpy(out,device_input+sum_input_size[count_layers], arr_neuron_size[count_layers-1] * sizeof(float),
                             cudaMemcpyDeviceToHost));

  return out;
}

GPUNetwork::~GPUNetwork() {
  delete num_blocks;
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

  checkErrorsCuda(cudaFree(test_device_dataset));
}
