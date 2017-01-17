#include "gpunetwork.cuh"

__global__
void tile_propagate_layer(){

int tid_x = blockIdx.x*blockDim.x+threadIdx.x;
//int tid_y = blockIdx.y*blockDim.y+threadIdx.y;

float output = 0;

for (unsigned int j = 0; j < net_input; j++) {
  output += net_neurons[tid_x].weights[j] * net_input[j];
}

output += net_neurons[tid_x].wbias;

net_neurons[tid_x]->output = 1 / (1 + exp(-output));
}

__global__
void tile_update_layer(){

int tid_x = blockIdx.x*blockDim.x+threadIdx.x;
//int tid_y = blockIdx.y*blockDim.y+threadIdx.y;

float dw;
  for (unsigned int i = 0; i < l->count_input; i++) {
    dw = learning_rate * l->input[i] * l->neurons[n]->delta;
    dw += momentum * l->neurons[n]->prvdeltas[i];
    l->neurons[tid_x]->prvdeltas[i] = dw;
    l->neurons[tid_x]->weights[i] += dw;
  }

}

__global__
void tile_layer_train(){

int tid_x = blockIdx.x*blockDim.x+threadIdx.x;
//int tid_y = blockIdx.y*blockDim.y+threadIdx.y;

float out;
float delta = 0;
for (unsigned int i = 0; i < pl->count_neurons; i++) {
  for (unsigned int j = neuron_start; j < neuron_end; j++) {
    delta += pl->neurons[i]->weights[j] * pl->neurons[i]->delta;
  }
}
for (unsigned int n = neuron_start; n < neuron_end; n++) {
  out = l->neurons[n]->output;

  l->neurons[n]->delta = out * (1 - out) * delta;
  l->neurons[n]->wbias += learning_rate * out * (1 - out) * delta;
}
}
