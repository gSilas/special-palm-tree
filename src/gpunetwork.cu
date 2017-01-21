#include "gpunetwork.cuh"

__global__ void
Device::tile_update_layer(float *device_input, float *device_weights,
                          float *device_delta, float *device_prvdeltas,
                          float learning_rate, float momentum,
                          unsigned int input_offset, unsigned int neuron_offset,
                          unsigned int input_size, unsigned int weight_offset) {

  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  // int tid_y = blockIdx.y*blockDim.y+threadIdx.y;

  float dw;
  for (unsigned int j = 0; j < input_size; j++) {
    dw = learning_rate * device_input[j + input_offset] *
         device_delta[neuron_offset + tid_x];
    dw += momentum * device_prvdeltas[j + weight_offset + tid_x * input_size];
    /*printf("threadIdx: %d dw: %f input %f delta %f \n", tid_x, dw,
           device_input[j + input_offset], device_delta[neuron_offset + tid_x]);
*/
    device_prvdeltas[j + weight_offset + tid_x * input_size] = dw;
    device_weights[j + weight_offset + tid_x * input_size] += dw;
  }
}

__global__ void Device::tile_propagate_layer(
    float *device_input, float *device_weights, float *device_wbias,
    float *device_output, unsigned int input_size, unsigned int neuron_size,
    unsigned int input_offset, unsigned int neuron_offset,
    unsigned int nl_neuron_offset, unsigned int weight_offset) {

  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  // int tid_y = blockIdx.y*blockDim.y+threadIdx.y;

  float output = 0;
  for (unsigned int j = 0; j < input_size; j++) {
    output += device_weights[j + weight_offset + tid_x * input_size] *
              device_input[j + input_offset];
  /*  printf(" prop threadIdx: %d we %f in %f input_off %d \n", tid_x,
           device_weights[j + weight_offset + tid_x * input_size],
           device_input[j + input_offset],input_offset); */
  }

  output += device_wbias[tid_x + neuron_offset];
  /*printf("threadIdx: %d input_offset: %d neuron_offset: %d\n", tid_x,
         input_offset, neuron_offset);*/
  float res = 1 / (1 + exp(-output));
  device_output[neuron_offset + tid_x] = res;
  device_input[nl_neuron_offset + tid_x] = res;
//  printf(" prop threadIdx: %d neuron_off %d \n", nl_neuron_offset);
}

__global__ void Device::tile_outlayer_train(float *device_delta,
                                            float *device_wbias,
                                            float *device_output,
                                            float *device_awaited_output,
                                            float learning_rate,
                                            unsigned int nl_neuron_offset) {

  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  // int tid_y = blockIdx.y*blockDim.y+threadIdx.y;
  float out;
  out = device_output[tid_x + nl_neuron_offset];

  device_delta[tid_x + nl_neuron_offset] =
      (device_awaited_output[tid_x] - out) * out * (1 - out);

  device_wbias[tid_x + nl_neuron_offset] +=
      learning_rate * (device_awaited_output[tid_x] - out) * out * (1 - out);
}

__global__ void Device::tile_layer_train(
    float *device_weights, float *device_wbias, float *device_delta,
    float *device_output, float *device_awaited_output, float learning_rate,
    unsigned int pl_neuron_size, unsigned int pl_input_size,
    unsigned int tl_weight_offset, unsigned int tl_neuron_offset,
    unsigned int nl_neuron_offset) {

  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  // int tid_y = blockIdx.y*blockDim.y+threadIdx.y;

  float out;
  float delta = 0;
  // nl = layer[this-1]| tl = layer[this] | pl = layer[this + 1]
  for (unsigned int i = 0; i < pl_neuron_size; i++)
    for (unsigned int j = 0; j < pl_input_size; j++) {
      delta += device_weights[tl_weight_offset + i * pl_input_size + j] *
               device_delta[tl_neuron_offset + i];
    }

  out = device_output[tid_x + nl_neuron_offset];
  /*printf("threadIdx: %d neuron_offset: %d layer_offset: %d\n", tid_x,
         neuron_offset, layer_offset);*/
  device_delta[tid_x + nl_neuron_offset] = out * (1 - out) * delta;
  device_wbias[tid_x + nl_neuron_offset] +=
      learning_rate * out * (1 - out) * delta;
}

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

  for (int l = 0; l < clayers; l++) {
    input_size += inputs[l];
    neuron_size += neurons[l];

    weight_size += inputs[l] * neurons[l];
    if (neurons[l] > device_props.maxThreadsPerBlock)
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

  for (int l = 1; l < clayers + 1; l++) {
    sum_input_size[l] = inputs[l - 1] + sum_input_size[l - 1];
    sum_neuron_size[l] = neurons[l - 1] + sum_neuron_size[l - 1];
    sum_weight_size[l] = neurons[l - 1] * inputs[l - 1];
  }

  for (int l = 0; l < clayers + 1; l++) {
    std::cout << l << " | " << sum_input_size[l] << " | " << sum_neuron_size[l]
              << std::endl;
  }

  std::cout << input_size << " | " << neuron_size << " | " << weight_size
            << std::endl
            << "cudaMalloc" << std::endl;

  checkErrorsCuda(cudaMalloc(&device_input, sizeof(float) * input_size));

  checkErrorsCuda(cudaMalloc(&device_wbias, sizeof(float) * neuron_size));
  checkErrorsCuda(cudaMalloc(&device_output, sizeof(float) * neuron_size));
  checkErrorsCuda(cudaMalloc(&device_delta, sizeof(float) * neuron_size));

  checkErrorsCuda(cudaMalloc(&device_weights, sizeof(float) * weight_size));
  checkErrorsCuda(cudaMalloc(&device_prvdeltas, sizeof(float) * weight_size));

  std::cout << "cudaMemset" << std::endl;

  checkErrorsCuda(cudaMemset(device_input, 0, sizeof(float) * input_size));

  checkErrorsCuda(cudaMemset(device_wbias, 1, sizeof(float) * neuron_size));
  checkErrorsCuda(cudaMemset(device_output, 1, sizeof(float) * neuron_size));
  checkErrorsCuda(cudaMemset(device_delta, 0, sizeof(float) * neuron_size));

  checkErrorsCuda(cudaMemset(device_weights, 1, sizeof(float) * weight_size));
  checkErrorsCuda(cudaMemset(device_prvdeltas, 0, sizeof(float) * weight_size));
}

void GPUNetwork::propagate_network(const float *input) {

  checkErrorsCuda(cudaMemcpy(device_input, input,
                             arr_input_size[0] * sizeof(float),
                             cudaMemcpyHostToDevice));

  for (unsigned int l = 1; l < count_layers +1; l++) {

    Device::tile_propagate_layer<<<num_blocks[l], threads_per_block[l]>>>(
        device_input, device_weights, device_wbias, device_output,
        arr_input_size[l-1], arr_neuron_size[l-1], sum_input_size[l-1],
        sum_neuron_size[l-1],sum_neuron_size[l], sum_weight_size[l-1]);

    /*std::cout << l << " | " << num_blocks[l] << " | " << threads_per_block[l]
              << " | " << arr_input_size[l] << " | " << arr_neuron_size[l]
              << " | " << sum_input_size[l] << " | " << sum_neuron_size[l]
              << " | " << sum_weight_size[l] << " | " << std::endl;*/

    cudaDeviceSynchronize();
  }
}

float GPUNetwork::train_network(const float *input, const float *awaited_output,
                                const float learning_rate, float momentum) {

  propagate_network(input);
  float *device_awaited_output;
  checkErrorsCuda(
      cudaMalloc(&device_awaited_output,
                 sizeof(float) * arr_neuron_size[count_layers - 1]));
  checkErrorsCuda(cudaMemcpy(device_awaited_output, awaited_output,
                             sizeof(float) * arr_neuron_size[count_layers - 1],
                             cudaMemcpyHostToDevice));

  float total_error = 0;
  float *out;
  out = getOutput();

  for (unsigned int i = 0; i < arr_neuron_size[count_layers - 1]; i++) {
    total_error +=
        0.5 * (awaited_output[i] - out[i]) * (awaited_output[i] - out[i]);
  }

  Device::tile_outlayer_train<<<num_blocks[count_layers - 1],
                                threads_per_block[count_layers - 1]>>>(
      device_delta, device_wbias, device_output, device_awaited_output,
      learning_rate, sum_neuron_size[count_layers - 1]);

  cudaDeviceSynchronize();

  for (int l = count_layers - 2; l >= 0; l--) {
    Device::tile_layer_train<<<num_blocks[l], threads_per_block[l]>>>(
        device_weights, device_wbias, device_delta, device_output,
        device_awaited_output, learning_rate, arr_neuron_size[l + 1],
        arr_input_size[l + 1], sum_weight_size[l + 1], sum_neuron_size[l + 1],
        sum_neuron_size[l]);

    cudaDeviceSynchronize();
    /*
        std::cout << " tile_layer_train " << std::endl;
        std::cout << l << " | " << num_blocks[l] << " | " <<
       threads_per_block[l]
                  << " | " << arr_neuron_size[l + 1] << " | "
                  << arr_input_size[l + 1] << " | " << sum_weight_size[l + 1]
                  << " | " << sum_neuron_size[l + 1] << " | " <<
       sum_neuron_size[l]
                  << std::endl; */
  }

  for (unsigned int l = 0; l < count_layers; l++) {
    Device::tile_update_layer<<<num_blocks[l], threads_per_block[l]>>>(
        device_input, device_weights, device_delta, device_prvdeltas,
        learning_rate, momentum, sum_input_size[l], sum_neuron_size[l],
        arr_input_size[l], sum_weight_size[l]);

    cudaDeviceSynchronize();
    /*
        std::cout << " tile_update_layer " << std::endl;
        std::cout << l << " | " << num_blocks[l] << " | " <<
       threads_per_block[l]
                  << " | " << sum_input_size[l] << " | " << sum_neuron_size[l]
                  << " | " << arr_input_size[l] << " | " << sum_weight_size[l]
                  << std::endl; */
  }

  delete out;
  // checkErrorsCuda(cudaFree(device_awaited_output));

  return total_error;
}

float *GPUNetwork::getOutput() {
  float *out = new float[sum_neuron_size[count_layers]];

  checkErrorsCuda(cudaMemcpy(out, device_output,
                             sum_neuron_size[count_layers] * sizeof(float),
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
  checkErrorsCuda(cudaFree(device_output));
}
