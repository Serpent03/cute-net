#pragma once
#include "common.h"
#include <stdbool.h>

/* ==== STRUCTURES & METHODS ==== */

typedef struct Neuron {
  float64 *weights; /* weights attached to the node. depends on the number of connections coming in. */
  uint32 num_weights;
  float64 bias;
  float64 value;
} Neuron;

typedef struct Layer {
  Neuron **neurons; /* list containing neurons */
  /* reserved for future..? */
} Layer;

typedef struct Training {
  uint32 iteration;
  float64 *loss; /* this stores all output neuron activation values. */
  float64 (*loss_function)(float64 output, float64 input_label);
  float64 (*derivative_function)(float64 value);
  float64 learning_rate;
  float64 **delta; /* this stores all the SGD deltas for easy access. */
} Training;

typedef struct Network {
  Layer **layers;
  uint32 num_layers;
  uint32 *num_neurons_per_layer; /* each i-th value dictates number of neurons in layer i */
  uint32 currLayerIdx; /* global pointer for the current layer being fore/backpropagated. */
  bool backprop;
  float64 (*activate)(float64 sum); /* this is the activation function. */
  Training *training;
} Network;

/**
 * @brief Leaky ReLU activation function.
 * @param value The non-activated value in question.
 * @return activated float64 value.
*/
float64 leakyRELU(float64 value);

/**
 * @brief Leaky RELU derivative function.
 * @param value The activated value in question.
 * @return derivative as a float64.
*/
float64 leakyRELU_d(float64 value);

/**
 * @brief Mean Square Error cost function.
 * @param output Output at the end of the network.
 * @param input_label The expected output fed with the training data.
 * @return mean squared error as a float64
*/
float64 meanSqErr(float64 output, float64 input_label); 

/**
 * @brief Initialize a neuron.
 * @param in_nodes The number of nuerons in the previous connecting to this neuron.
 * @return A neuron object.
*/
Neuron *init_neuron(uint32 in_nodes);

/**
 * @brief Initialize a layer.
 * @param num_neurons Number of neurons in this layer.
 * @param in_nodes The number of neurons in the previous layer connected to each neuron in this layer.
 * @return A layer object.
*/
Layer *init_layer(uint32 num_neurons, uint32 in_nodes);

/**
 * @brief Initialize a neural network.
 * @param num_neurons_per_layer The number of neurons per layer in the network, passed as an array.
  The num_neurons_per_layer array must be passed via reference(pointer) and should contain information in the 
  format [ 2, 4, 4, 1 ], which translates to:
  - 2 input nodes, 
  - 4 neurons in the first hidden layer, 4 neurons in the second hidden layer,
  - 1 neuron in the output layer.
 * @param num_layers The total number of layers, including the input, hidden and output layers.
 * @return A network object.
*/
Network *init_network(uint32 *num_neurons_per_layer, uint32 num_layers);

/**
 * @brief Forward propagation inside a neural network. Takes the data 
  from the input layer to the output layer.
 * @param network The neural network to propagate.
 * @return False if the forward propagation has been completed to the output layer.
*/
void forward_propagate(Network *network);

/**
 * @brief Backward propagation inside a neural network
 * @param network The neural network to propagate.
 * @return False if the backward propagation has been completed to the input layer.
*/
void backward_propagate(Network *network, float64 *label_data);

/**
 * @brief Test the network based on data.
 * @param data The data to be fed.
 * @param len The dimensionality of the data.
 * @param network The neural network to test.
 * @return A float64 array containing data from the output layer.
*/
float64 *test_network(float64 *data, uint32 len, Network *network);

/**
 * @brief Train the network on a specific dataset.
 * @param network The neural network to train.
 * @param training_data A 2-dimensional array representing the input data.
 * @param training_data_len The number of inputs to be fed.
 * @param batch_len The amount of training_data arrays.
 * @param label_data A 2-dimensional array representing the expected output data.
 * @param label_data_len The number of outputs to be expected.
 * @param epoch The number of times to train on a single piece of data.
*/
void train_network(Network *network, float64 **training_data, uint32 training_data_len, uint32 batch_len, float64 **label_data, uint32 label_data_len, uint32 epoch);

/*
  During forward propagation, the workflow will be as such:
  >> Value gets inside <<Neuron>> after being activated.
  >> Check the <<Layer>> structure; get current_layer + 1
  >> Pick the first <<Neuron>> in the next layer. Iterate through the <<weights>> list
  >> For each index i of the <<weights>> list, multiply it by the ith <<Neuron>> in the current <<Layer>>
  >> Get the summation, and apply activation functions, and then put it in the <<Neuron>> in the next layer.
*/

void populate_input(float64 *data, uint32 len, Network *network);
void debug_network(Network *network);