#ifndef PARALLEL_H
#define PARALLEL_H

#include <thread>
#include <vector>

#include "layer.h"

namespace Parallel {

void tile_propagate_layer(Layer *l, int neuron_start, int neuron_end);

void tile_layer_train(Layer *l, Layer *pl, int neuron_start, int neuron_end,
                      float learning_rate);

void tile_layer_update(Layer *l, int neuron_start, int neuron_end,
                       float learning_rate, float momentum);
}

#endif
