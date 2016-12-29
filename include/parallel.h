#ifndef PARALLEL_H
#define PARALLEL_H

#include <thread>
#include <vector>

#include "layer.h"

namespace Parallel {
void tile_propagate_layer(Layer *l, int neuron_start);
}

#endif
