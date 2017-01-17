#include "gpunetwork.h"

void GPUNetwork::init_network(unsigned int *inputs, unsigned int *neurons,
                              unsigned int clayers) {

  count_layers = clayers;

  layers = new Layer *[clayers];

  for (unsigned int l = 0; l < clayers; l++) {
    net_layers[l] = new Layer;
    net_layers[l]->init_layer(inputs[l], neurons[l]);
  }
  /*
    for (unsigned int l = 0; l < clayers; l++) {
      gpuErrchk(cudaMalloc(&net_inputs[l], sizeof(float) * inputs[l]));
      neuron_count += neurons[l];
    }

    Neuron tmp[neuron_count];
    for (unsigned int l = 0; l < neuron_count; l++) {
      tmp[l] = Neuron;
      tmp[l].init_neuron();
    }

    gpuErrchk(cudaMalloc(&net_neurons, sizeof(Neuron) * neuron_count));
    gpuErrchk(cudaMemcpy(&net_neurons, &tmp, sizeof(Neuron) * neuron_count,
                         cudaMemcpyHostToDevice));
                         */
  cudaDeviceProp device_props;
  gpuErrchk(cudaGetDeviceProperties(&device_props, 0));
  unsigned int max_threads_per_block = device_props.maxThreadsPerBlock;

  unsigned int max_threads_per_block_sqrt = std::sqrt(max_threads_per_block);
  assert(max_threads_per_block_sqrt * max_threads_per_block_sqrt ==
         max_threads_per_block);
  dim3 num_threads_per_block(std::min(n_rows, max_threads_per_block_sqrt),
                             std::min(n_cols, max_threads_per_block_sqrt));
  dim3 num_blocks(n_rows / num_threads_per_block.x,
                  n_cols / num_threads_per_block.y);
  if (0 == num_blocks.x) {
    num_blocks.x++;
  }
  if (0 == num_blocks.y) {
    num_blocks.y++;
  }
  std::cout << "num_blocks = " << num_blocks.x << " / " << num_blocks.y
            << std::endl;
  std::cout << "num_threads_per_block = " << num_threads_per_block.x << " / "
            << num_threads_per_block.y << std::endl;

  for (unsigned int l = 0; l < clayers; l++) {
    gpuErrchk(cudaMalloc(&device_layers[l], sizeof(net_layers[l])));
    gpuErrchk(cudaMemcpy(&device_layers[l], &net_layers[l],
                         sizeof(net_layers[l]), cudaMemcpyHostToDevice));
  }
}

void GPUNetwork::propagate_network(const float *input) {

  gpuErrchk(cudaMemcpy(&device_layers[l].input, &input,
                       sizeof(float) * inputs[i], cudaMemcpyHostToDevice));

  for (unsigned int l = 1; l < count_layers; l++) {

    Device::tile_propagate_layer<<<num_blocks, num_threads_per_block>>>(
        device_layers[l]);

    if (l < count_layers - 1) {
      for (unsigned int n = 0; n < layers[l]->count_neurons; n++) {
        layers[l + 1]->input[n] = layers[l]->neurons[n]->output;
      }
    }
  }
}

float GPUNetwork::train_network(const float *input, const float *awaited_output,
                                const float learning_rate, float momentum) {

  propagate_network(input);

  float total_error = 0;
  float out;

  Layer *output_layer = layers[count_layers - 1];

  for (unsigned int i = 0; i < output_layer->count_neurons; i++) {
    out = output_layer->neurons[i]->output;

    total_error += 0.5 * (awaited_output[i] - out) * (awaited_output[i] - out);

    output_layer->neurons[i]->delta =
        (awaited_output[i] - out) * out * (1 - out);

    output_layer->neurons[i]->wbias +=
        learning_rate * (awaited_output[i] - out) * out * (1 - out);
  }

  Device::tile_layer_train<<<num_blocks, num_threads_per_block>>>(
      layers[l], layers[l + 1], tiling[i], tiling[i + 1], learning_rate);

  for (unsigned int l = 0; l < count_layers; l++) {
    Device::tile_layer_update<<<num_blocks, num_threads_per_block>>>(
        layers[l], tiling[i], tiling[i + 1], learning_rate, momentum);
  }

  return total_error;
}

GPUNetwork::~GPUNetwork() {
  for (unsigned int i = 0; i < count_layers; i++) {
    delete layers[i];
  }
  delete layers;
}
