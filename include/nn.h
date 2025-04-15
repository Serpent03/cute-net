#pragma once
#include "common.h"
#include <stdbool.h>

#define ACTIVATION_SIGMOID 0
#define ACTIVATION_RELU 1

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
  float32 learning_rate;
  float64 **delta; /* this stores all the SGD deltas for easy access. */
} Training;

typedef struct Network {
  Layer **layers;
  uint32 num_layers;
  uint32 *num_neurons_per_layer; /* each i-th value dictates number of neurons in layer i */
  uint32 currLayerIdx; /* global pointer for the current layer being fore/backpropagated. */
  uint32 activation_type;
  float64 (*activate)(float64 sum); /* this is the activation function. */
  Training *training;
} Network;

/**
 * @brief Initialize a neural network.
 * @param num_neurons_per_layer The number of neurons per layer in the network, passed as an array.
  The num_neurons_per_layer array must be passed via reference(pointer) and should contain information in the 
  format [ 2, 4, 4, 1 ], which translates to:
  - 2 input nodes, 
  - 4 neurons in the first hidden layer, 4 neurons in the second hidden layer,
  - 1 neuron in the output layer.
 * @param num_layers The total number of layers, including the input, hidden and output layers.
 * @param learning_rate Hyperparameter to set the learning rate of the network during backpropagation.
 * @return A network object.
*/
Network *init_network(uint32 *num_neurons_per_layer, uint32 num_layers, uint8 activation_type, float32 learning_rate);

/**
 * @brief Save a network to file.
 * @param n The network in question.
 * @param file The name of the file to save the network to.
*/
void save_network(Network *n, char *file);

/**
 * @brief Initialize a network from file.
 * @param file The name of the file containing the network data.
 * @return An initialized neural network.
*/
Network *load_network(char *file);

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

/**
 * @brief Free the network N.
*/
void free_network(Network *n);
void debug_network(Network *network);

