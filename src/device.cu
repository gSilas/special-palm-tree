#include "device.cuh"

__global__ void
Device::tile_update_layer(float *device_input, float *device_weights,
                          float *device_delta, float *device_prvdeltas,
                          float learning_rate, float momentum,
                          unsigned int input_offset, unsigned int neuron_offset,
                          unsigned int input_size, unsigned int weight_offset) {

  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;

  float dw;
  for (unsigned int j = 0; j < input_size; j++) {

    unsigned int index = j + weight_offset + tid_x * input_size;

    dw = learning_rate * device_input[j + input_offset] *
         device_delta[neuron_offset + tid_x];
    dw += momentum * device_prvdeltas[index];

    device_prvdeltas[index] = dw;
    device_weights[index] += dw;
  }
}
__global__ void Device::tile_propagate_inlayer(
    float *data_set, float *device_input, float *device_weights,
    float *device_wbias, unsigned int input_size, unsigned int neuron_size) {

  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;

  float output = 0;
  for (unsigned int j = 0; j < input_size; j++) {
    device_input[j] = data_set[j];
    output += device_weights[j + tid_x * input_size] * device_input[j];
  }

  output += device_wbias[tid_x];

  float res = 1 / (1 + exp(-output));
  device_input[input_size + tid_x] = res;
}

__global__ void Device::tile_propagate_layer(
    float *device_input, float *device_weights, float *device_wbias,
    unsigned int input_size, unsigned int input_offset,
    unsigned int neuron_offset, unsigned int output_offset,
    unsigned int weight_offset) {

  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;

  unsigned int index = neuron_offset + tid_x;

  float output = 0;
  for (unsigned int j = 0; j < input_size; j++) {

    output += device_weights[j + weight_offset + tid_x * input_size] *
              device_input[j + input_offset];
  }

  output += device_wbias[input_size + tid_x];

  float res = 1 / (1 + exp(-output));
  device_input[output_offset + tid_x] = res;
}

__global__ void
Device::tile_outlayer_train(float *device_input, float *device_delta,
                            float *device_wbias, float *device_awaited_output,
                            float learning_rate, unsigned int output_offset,
                            unsigned int nl_neuron_offset) {

  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;

  unsigned int index = nl_neuron_offset + tid_x;
  float out;

  out = device_input[output_offset + tid_x];
  float delta = (device_awaited_output[tid_x] - out) * out * (1 - out);
  device_delta[index] = delta;
  device_wbias[index] += learning_rate * delta;
}

__global__ void Device::tile_layer_train(
    float *device_input, float *device_weights, float *device_wbias,
    float *device_delta, float *device_awaited_output, float learning_rate,
    unsigned int pl_neuron_size, unsigned int pl_input_size,
    unsigned int tl_weight_offset, unsigned int tl_neuron_offset,
    unsigned int tl_input_size, unsigned int nl_neuron_offset) {

  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;

  unsigned int index = nl_neuron_offset + tid_x;

  float out;
  float delta = 0;

  for (unsigned int i = 0; i < pl_neuron_size; i++) {
    for (unsigned int j = 0; j < pl_input_size; j++) {
      delta += device_weights[tl_weight_offset + i * pl_input_size + j] *
               device_delta[tl_neuron_offset + i];
    }
  }

  out = device_input[index + tl_input_size];

  float rdelta = out * (1 - out) * delta;
  device_delta[index] = rdelta;
  device_wbias[index] += learning_rate * rdelta;
}
