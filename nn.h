#pragma once

/* ==== DEFINITIONS & TYPES ==== */

#define IN_NODES 2
#define OUT_NODES 1

typedef unsigned char uint8;
typedef char int8;

typedef unsigned short int uint16;
typedef short int int16;

typedef unsigned int uint32;
typedef int int32;

typedef float float32;
typedef double float64;

/* ==== STRUCTURES & METHODS ==== */

typedef struct Neuron {
  float64 *weights; /* weights attached to the node. depends on the number of connections coming in. */
  uint32 num_weights;
  float64 bias;
  float64 value;
  /* add a function pointer for activation function */
} Neuron;

typedef struct Layer {
  Neuron **neurons; /* list containing neurons */
  /* reserved for future..? */
} Layer;

typedef struct Network {
  Layer **layers;
  uint32 num_layers;
  uint32 *num_neurons_per_layer; /* each i-th value dictates number of neurons in layer i */
} Network;

/* there has to be a function that takes in the width x breadth of the entire
network and returns the network*/

/*
  @brief Initialize a neuron.
  @param in_nodes The number of nuerons in the previous connecting to this neuron.
  @return A neuron object.
*/
Neuron *init_neuron(uint32 in_nodes);

/*
  @brief Initialize a layer.
  @param num_neurons Number of neurons in this layer.
  @param in_nodes The number of neurons in the previous layer connected to each neuron in this layer.
  @return A layer object.
*/
Layer *init_layer(uint32 num_neurons, uint32 in_nodes);

/*
  @brief Initialize a neural network.
  @param num_neurons_per_layer The number of neurons per layer in the network, passed as an array.
  The num_neurons_per_layer array must be passed via reference(pointer) and should contain information in the 
  format [ 2, 4, 4, 1 ], which translates to: 
  - 2 input nodes, 
  - 4 neurons in the first hidden layer, 4 neurons in the second hidden layer,
  - 1 neuron in the output layer.
  @param num_layers The total number of layers, including the input, hidden and output layers.
  @return A network object.
*/
Network *init_network(uint32 *num_neurons_per_layer, uint32 num_layers);

/*
  During forward propagation, the workflow will be as such:
  >> Value gets inside <<Neuron>> after being activated.
  >> Check the <<Layer>> structure; get current_layer + 1
  >> Pick the first <<Neuron>> in the next layer. Iterate through the <<weights>> list
  >> For each index i of the <<weights>> list, multiply it by the ith <<Neuron>> in the current <<Layer>>
  >> Get the summation, and apply activation functions, and then put it in the <<Neuron>> in the next layer.
*/