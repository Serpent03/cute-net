#include "nn.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/*
  New idea: instead of representing layers with structures
  how about representing with arrays? This way, CUDA integration can also 
  be done in a cleaner fashion!

  This can be represented with a matrix multiplication for simplicity.

  List    List      List
  Input1  Neuron1_1 Neuron2_1
  Input2  Neuron1_2 Neuron2_2
  Input3  Neuron1_3 Neuron2_3

*/

/* 
  Let's get forward feed working first

  List    List      List
  Input1  Neuron1_1 Neuron2_1
  Input2  Neuron1_2
*/


Neuron *init_neuron(uint32 in_nodes) {
  /* in_nodes is determined by the number of neurons in the 
  previous layer - as they are all interconnected. */
  Neuron *n = (Neuron*)calloc(1, sizeof(Neuron));

  n->weights = (float64*)calloc(in_nodes, sizeof(float64));
  n->num_weights = in_nodes;
  n->bias = 0;
  n->value = 0;

  for (uint32 i = 0; i < n->num_weights; i++) {
    n->weights[i] = ((float64)(rand() % 100)) / 100;
    /* randomize a value between 0 and 1. Will need a better system */
  }
  return n;
}

Layer *init_layer(uint32 num_neurons, uint32 in_nodes) {
  Layer *l = (Layer*)calloc(1, sizeof(Layer));
  l->neurons = (Neuron**)calloc(num_neurons, sizeof(Neuron*));
  for (uint32 i = 0; i < num_neurons; i++) {
    l->neurons[i] = init_neuron(in_nodes);
  }
  return l;
}

Network *init_network(uint32 *num_neurons_per_layer, uint32 num_layers) {
  Network *n = (Network*)calloc(1, sizeof(Network));
  n->num_layers = num_layers;
  n->num_neurons_per_layer = (uint32*)calloc(n->num_layers, sizeof(uint32));
  memcpy(n->num_neurons_per_layer, num_neurons_per_layer, num_layers * sizeof(uint32));
  n->layers = (Layer**)calloc(n->num_layers, sizeof(Layer*));

  /* the 0th layer is treated as the input layer. */
  n->layers[0] = init_layer(n->num_neurons_per_layer[0], 0);
  for (uint32 layer_ptr = 1; layer_ptr < n->num_layers; layer_ptr++) {
    n->layers[layer_ptr] = init_layer(n->num_neurons_per_layer[layer_ptr], n->num_neurons_per_layer[layer_ptr - 1]);
  }
  return n;
}