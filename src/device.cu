#include "device.cuh"

__global__ void
Device::tile_update_layer(float *device_input, float *device_weights,
                          float *device_delta, float *device_prvdeltas,
                          float learning_rate, float momentum,
                          unsigned int input_size, unsigned int neuron_size) {

  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;

  float dw;
  for (unsigned int j = 0; j < input_size; j++) {

    unsigned int index = j + tid_x * input_size;

    dw = learning_rate * device_input[j] * device_delta[tid_x];
    dw += momentum * device_prvdeltas[index];

    device_prvdeltas[index] = dw;
    device_weights[index] += dw;
  }
}
__global__ void
Device::tile_propagate_inlayer(float *data_set, float *device_input,
                               float *nl_device_input, float *device_weights,
                               float *device_wbias, unsigned int input_size,
                               unsigned int neuron_size) {

  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;

  float output = 0;
  for (unsigned int j = 0; j < input_size; j++) {
    device_input[j] = data_set[j];
    output += device_weights[j + tid_x * input_size] * device_input[j];
  }

  output += device_wbias[tid_x];

  float res = 1 / (1 + exp(-output));
  nl_device_input[tid_x] = res;
}

__global__ void Device::tile_propagate_layer(
    float *device_input, float *nl_device_input, float *device_weights,
    float *device_wbias, unsigned int input_size, unsigned int neuron_size) {

  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;

  float output = 0;
  for (unsigned int j = 0; j < input_size; j++) {

    output += device_weights[j + tid_x * input_size] * device_input[j];
  }

  output += device_wbias[tid_x];

  float res = 1 / (1 + exp(-output));
  nl_device_input[tid_x] = res;
}

__global__ void Device::tile_outlayer_train(float *device_output,
                                            float *device_delta,
                                            float *device_wbias,
                                            float *device_awaited_output,
                                            float learning_rate) {

  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;

  float out = device_output[tid_x];

  float delta = (device_awaited_output[tid_x] - out) * out * (1 - out);
  device_delta[tid_x] = delta;
  device_wbias[tid_x] += learning_rate * delta;
}

__global__ void
Device::tile_layer_train(float *device_output, float *pl_device_weights,
                         float *pl_device_delta, float *device_wbias,
                         float *device_delta, float *device_awaited_output,
                         float learning_rate, unsigned int pl_input_size,
                         unsigned int pl_neuron_size) {

  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;

  float out;
  float delta = 0;

  for (unsigned int i = 0; i < pl_neuron_size; i++) {
    for (unsigned int j = 0; j < pl_input_size; j++) {
      delta += pl_device_weights[i * pl_input_size + j] * pl_device_delta[i];
    }
  }

  out = device_output[tid_x];

  float rdelta = out * (1 - out) * delta;
  device_delta[tid_x] = rdelta;
  device_wbias[tid_x] += learning_rate * rdelta;
}
