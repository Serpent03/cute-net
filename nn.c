#include "nn.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>

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

Neuron *init_neuron(uint32 in_nodes) {
  /* in_nodes is determined by the number of neurons in the 
  previous layer - as they are all interconnected. */
  Neuron *n = (Neuron*)calloc(1, sizeof(Neuron));

  n->weights = (float64*)calloc(in_nodes, sizeof(float64));
  n->num_weights = in_nodes;
  n->bias = 0;
  n->value = 0;

  for (uint32 i = 0; i < n->num_weights; i++) {
    float64 wval = ((float64)(rand() % 200 - 100)) / 100; /* random value between -1 and 1 */
    n->weights[i] = wval;
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

Training *init_training(uint32 output_layer_size) {
  Training *t = (Training*)calloc(1, sizeof(Training));
  t->iteration = 0;
  t->loss = (float64*)calloc(output_layer_size, sizeof(float64));
  t->loss_function = &meanSqErr;
  t->learning_rate = 0.001;
  return t;
}

Network *init_network(uint32 *num_neurons_per_layer, uint32 num_layers) {
  srand(time(NULL)); /* At the start of the setup. */

  Network *n = (Network*)calloc(1, sizeof(Network));
  n->num_layers = num_layers;
  n->num_neurons_per_layer = (uint32*)calloc(n->num_layers, sizeof(uint32));
  memcpy(n->num_neurons_per_layer, num_neurons_per_layer, num_layers * sizeof(uint32));
  n->layers = (Layer**)calloc(n->num_layers, sizeof(Layer*));
  n->currLayerIdx = 0;
  n->activate = &leakyRELU;
  n->training = init_training(n->num_neurons_per_layer[n->num_layers - 1]);

  /* the 0th layer is treated as the input layer. */
  n->layers[0] = init_layer(n->num_neurons_per_layer[0], 0);
  for (uint32 layer_ptr = 1; layer_ptr < n->num_layers; layer_ptr++) {
    n->layers[layer_ptr] = init_layer(n->num_neurons_per_layer[layer_ptr], n->num_neurons_per_layer[layer_ptr - 1]);
  }
  return n;
}

float64 leakyRELU(float64 value) {
  /* ReLU is pretty much max(0, value). if we get negative values, however, we can quickly get
  stuck. In that scenario, we simply do max(value * -0.001, value). If the value is positive, then we
  choose it, otherwise we reduce its magnitude and change the direction, and take it. */
  float64 activation = value > value * 0.001 ? value : value * 0.001;
  return activation;
}

float64 meanSqErr(float64 output, float64 input_label) {
  return (float64)pow(output - input_label, 2);
}

void debug_network(Network *network) { 
  for (uint32 i = 0; i < network->num_layers; i++) {
    for (uint32 j = 0; j < network->num_neurons_per_layer[i]; j++) {
      Neuron *curr = network->layers[i]->neurons[j];
      printf("| V: %f, b: %f [", curr->value, curr->bias);
      for (uint32 k = 0; k < curr->num_weights; k++) {
        printf(" ");
        printf("%f", curr->weights[k]);
      }
      printf(" ] |");
    }
    printf("\n");
  }
  printf("\n");
}

/* 
  Let's get forward feed working first

  List    List      List
  Input1  Neuron1_1 Neuron2_1
  Input2  Neuron1_2
*/

/* 
  The workflow is as follows:
  >> The input <<Neuron.value>> gets populated as the input.
  >> In the next layer, we choose the first <<Neuron>>. With each index of weight present on it,
  >> we take the i-th <<Neuron>> in the current layer, sum up the values taken from the <<Neuron.value>> 
  >> field in the current layer, and put the activation inside the <<Neuron.value>> field in the next layer.
  >> The same process is repeated for the second neuron in the second layer, and then this keeps happening
  >> until we get to the output layer, where a MSE/etc are calculated for the cost analysis.
*/

bool forward_propagate(Network *network) {
  if (network->currLayerIdx >= network->num_layers - 1) {
    /* at this point, the result is available at the output layer.
    we can fetch the activated result and move on with cost analysis. */
    return false;
  }
  uint32 currLayerIdx = network->currLayerIdx;
  network->currLayerIdx++;

  for (uint32 i = 0; i < network->num_neurons_per_layer[currLayerIdx + 1]; i++) {
    /* iterate through all neurons in the next layer. */
    float64 sum = 0;
    Neuron *nextNeuron = network->layers[currLayerIdx + 1]->neurons[i];
    for (uint32 j = 0; j < network->num_neurons_per_layer[currLayerIdx]; j++) {
      /* get activated result of every neuron from the current layer, and then
      sum it with the weights with the selected neuron in the next layer.
      for each weight in the nextNeuron, we get the corresponding currNeuron activated data to be summed */
      Neuron *currNeuron = network->layers[currLayerIdx]->neurons[j];
      sum += currNeuron->value * nextNeuron->weights[j];
    }
    sum += nextNeuron->bias;
    nextNeuron->value = network->activate(sum); /* store the activated sum inside the next neuron */
  }
  // debug_network(network);
  return true;
}

bool backward_propagate(Network *network) {
  if (network->currLayerIdx <= 0) {
    return false;
  }
  /* compute the loss and do some magic stuff here to influence weights.. */

  return true;
}

void train_network(Network *network, float64 **training_data, uint32 training_data_len, uint32 training_data_batch_len, float64 **label_data, uint32 label_data_batch_len, uint32 label_data_len) {
  /* the training_data is a 2D array:
    - There are training_data_batch_len number of batches
    - Each batch has training_data_len number of inputs to be fed
  */
  for (uint32 i = 0; i < training_data_batch_len; i++) {
    float64 *retdata = test_network(training_data[i], training_data_len, network);
    network->currLayerIdx = network->num_layers - 1;
    memcpy(network->training->loss, retdata, network->currLayerIdx * sizeof(float64));
    free(retdata); /* <== this is important! if we don't free it, we're looking at a LOT of memory leakage. */
    backward_propagate(network);
  }
}

float64 *test_network(float64 *data, uint32 len, Network *network) {
  populate_input(data, len, network);
  while (forward_propagate(network));
  float64 *retdata = (float64*)malloc(network->num_neurons_per_layer[network->currLayerIdx] * sizeof(float64));
  for (uint32 i = 0; i < network->num_neurons_per_layer[network->currLayerIdx]; i++) {
    retdata[i] = network->layers[network->currLayerIdx]->neurons[i]->value;
  }
  network->currLayerIdx = 0;
  return retdata;
}

void populate_input(float64 *data, uint32 len, Network *network) {
  /* Make sure that the input data has the same dimensionality as the input layer. */
  assert(len == network->num_neurons_per_layer[0]);
  Layer *input_layer = network->layers[0];
  for (uint32 i = 0; i < len; i++) {
    input_layer->neurons[i]->value = data[i];
  }
}

/*
  Now to test out the network with custom weights/values. Once that is done, we can shift forward
  and write the backpropagation logic, with the gradient descent and weighing in the neurons -- which is really where the meat of the matter is. Once that is done, we simply have to tie it all together into a testing/training
  tool.
*/