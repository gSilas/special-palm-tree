#include "gpunetwork.cuh"

__global__
void tile_propagate_layer(float* in){

int tid_x = blockIdx.x*blockDim.x+threadIdx.x;
//int tid_y = blockIdx.y*blockDim.y+threadIdx.y;

float output = 0;

for (unsigned int j = 0; j < net_input; j++) {
  output += net_neurons[tid_x].weights[j] * net_input[j];
}

output += net_neurons[i].wbias;

net_neurons[tid_x]->output = 1 / (1 + exp(-output));
}

__global__
void tile_update_layer(){

int tid_x = blockIdx.x*blockDim.x+threadIdx.x;
//int tid_y = blockIdx.y*blockDim.y+threadIdx.y;


}

__global__
void tile_layer_train(){

int tid_x = blockIdx.x*blockDim.x+threadIdx.x;
//int tid_y = blockIdx.y*blockDim.y+threadIdx.y;


}
