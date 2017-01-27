#include "device.cuh"

__global__ void Device::set_dataset(float *device_input, float *data_set,
                                    unsigned int input_size) {
  unsigned int tid_x = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid_x < input_size) {

    device_input[tid_x] = data_set[tid_x];
    // printf("threadIdx %d set %f\n", tid_x, data_set[tid_x]);
  }
}

__global__ void Device::set_layer_memory(float *device_delta,
                                         float *device_prvdeltas,
                                         unsigned int input_size,
                                         unsigned int neuron_size) {

  unsigned int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid_x < neuron_size) {

    device_delta[tid_x] = 0.0f;

    unsigned int index = tid_x * input_size;

    for (unsigned int i = 0; i < input_size; i++) {
      device_prvdeltas[i + index] = 0.0f;
    }
  }
}

__global__ void
Device::tile_update_layer(float *device_input, float *device_weights,
                          float *device_delta, float *device_prvdeltas,
                          float learning_rate, float momentum,
                          unsigned int input_size, unsigned int neuron_size) {

  unsigned int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid_x < neuron_size * input_size) {

    float dw;

    dw = learning_rate * device_input[tid_x % input_size] *
         device_delta[(int)floorf((float)tid_x / (float)input_size)];

    dw += momentum * device_prvdeltas[tid_x];

    device_prvdeltas[tid_x] = dw;
    device_weights[tid_x] += dw;
  }
}
__global__ void Device::tile_propagate_inlayer(
    float *device_input, float *nl_device_input, float *device_weights,
    float *device_wbias, unsigned int input_size, unsigned int neuron_size) {

  unsigned int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid_x < neuron_size) {

    float output = 0;
    unsigned int index = tid_x * input_size;

    for (unsigned int j = 0; j < input_size; j++) {
      output += device_weights[j + index] * device_input[j];
    }

    output += device_wbias[tid_x];

    float res = 1 / (1 + expf(-output));
    nl_device_input[tid_x] = res;
  }
}

__global__ void Device::tile_propagate_layer(
    float *device_input, float *nl_device_input, float *device_weights,
    float *device_wbias, unsigned int input_size, unsigned int neuron_size) {

  unsigned int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid_x < neuron_size) {

    float output = 0;
    unsigned int index = tid_x * input_size;

    for (unsigned int j = 0; j < input_size; j++) {

      output += device_weights[j + index] * device_input[j];
    }

    output += device_wbias[tid_x];

    float res = 1 / (1 + expf(-output));
    nl_device_input[tid_x] = res;
  }
}

__global__ void
Device::tile_outlayer_train(float *device_output, float *device_delta,
                            float *device_wbias, float *device_awaited_output,
                            float learning_rate, unsigned int input_size,
                            unsigned int neuron_size) {

  unsigned int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid_x < neuron_size) {

    float out = device_output[tid_x];

    float delta = (device_awaited_output[tid_x] - out) * out * (1 - out);
    device_delta[tid_x] = delta;
    device_wbias[tid_x] += learning_rate * delta;
  }
}

__inline__ __device__
float warpReduceSum(float val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2)
    val += __shfl_down(val, offset);
  return val;
}

__inline__ __device__
float blockReduceSum(float val) {

  static __shared__ float shared[32]; // Shared mem for 32 partial sums
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceSum(val);     // Each warp performs partial reduction

  if (lane==0) shared[wid]=val; // Write reduced value to shared memory

  __syncthreads();              // Wait for all partial reductions

  //read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

  if (wid==0) val = warpReduceSum(val); //Final reduce within first warp

  return val;
}

__global__ void Device::reduction(float *data, float* out_data, unsigned int size) {
  float sum = 0;
  // reduce multiple elements per thread
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
    sum += data[i];
  }
  sum = blockReduceSum(sum);
  if (threadIdx.x == 0)
    out_data[blockIdx.x] = sum;
}

__global__ void Device::tile_layer_delta(float *device_delta_summands,
                                         float *pl_device_weights,
                                         float *pl_device_delta,
                                         unsigned int input_size,
                                         unsigned int neuron_size) {
  unsigned int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid_x < neuron_size * input_size) {
    device_delta_summands[tid_x] =
        pl_device_weights[tid_x] *
        pl_device_delta[(int)floorf((float)tid_x / (float)input_size)];
  }
}

__global__ void Device::tile_layer_train(
    float *device_output, float *device_delta_summands, float *device_wbias,
    float *device_delta, float *device_awaited_output, float learning_rate,
    unsigned int pl_input_size, unsigned int pl_neuron_size,
    unsigned int input_size, unsigned int neuron_size) {

  unsigned int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid_x < neuron_size) {

    float out;
    /*float delta = 0;

    for (unsigned int i = 0; i < pl_neuron_size; i++) {
      for (unsigned int j = 0; j < pl_input_size; j++) {
        delta += pl_device_weights[i * pl_input_size + j] * pl_device_delta[i];
      }
    }*/
    float delta = device_delta_summands[0];

    out = device_output[tid_x];

    float rdelta = out * (1 - out) * delta;
    device_delta[tid_x] = rdelta;
    device_wbias[tid_x] += learning_rate * rdelta;
  }
}
