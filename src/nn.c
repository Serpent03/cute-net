#include "../include/nn.h"
#include "../include/activation.h"
#include "../include/fileops.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>

FILE *fptr; /* this pointer interacts directly with the weights file. */

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
  Neuron *n = (Neuron*)malloc(1 * sizeof(Neuron));

  n->weights = (float64*)malloc(in_nodes * sizeof(float64));
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
  Layer *l = (Layer*)malloc(1 * sizeof(Layer));
  l->neurons = (Neuron**)malloc(num_neurons * sizeof(Neuron*));
  for (uint32 i = 0; i < num_neurons; i++) {
    l->neurons[i] = init_neuron(in_nodes);
  }
  return l;
}

Training *init_training(uint32 output_layer_size, uint32 num_layers, uint8 activation_type, float64 learning_rate) {
  Training *t = (Training*)malloc(1 * sizeof(Training));
  t->iteration = 0;
  t->loss = (float64*)malloc(output_layer_size * sizeof(float64));
  t->loss_function = &meanSqErr;
  t->learning_rate = learning_rate; /* 0.1 seems to be good for now..? */
  switch (activation_type) {
    case ACTIVATION_RELU:
      t->derivative_function = &leakyRELU_d;
      break;
    case ACTIVATION_SIGMOID:
      t->derivative_function = &sigmoid_d;
      break;
    default:
      assert(false);
  }
  t->delta = (float64**)malloc(num_layers * sizeof(float64*));
  return t;
}

Network *init_network(uint32 *num_neurons_per_layer, uint32 num_layers, uint8 activation_type, float64 learning_rate) {
  srand(time(NULL)); /* At the start of the setup. */
  Network *n = (Network*)malloc(1 * sizeof(Network));
  n->num_layers = num_layers;
  n->num_neurons_per_layer = (uint32*)malloc(n->num_layers * sizeof(uint32));
  memcpy(n->num_neurons_per_layer, num_neurons_per_layer, num_layers * sizeof(uint32));
  n->layers = (Layer**)malloc(n->num_layers * sizeof(Layer*));
  n->currLayerIdx = 0;

  switch (activation_type) {
    case ACTIVATION_RELU:
      n->activate = &leakyRELU;
      break;
    case ACTIVATION_SIGMOID:
      n->activate = &sigmoid;
      break;
    default:
      assert(false);
  }

  n->training = init_training(n->num_neurons_per_layer[n->num_layers - 1], n->num_layers, activation_type, learning_rate);

  /* the 0th layer is treated as the input layer. */
  n->layers[0] = init_layer(n->num_neurons_per_layer[0], 0);
  n->training->delta[0] = (float64*)malloc(n->num_neurons_per_layer[0] * sizeof(float64)); /* not sure if I will end up needing this. */
  for (uint32 layer_ptr = 1; layer_ptr < n->num_layers; layer_ptr++) {
    n->layers[layer_ptr] = init_layer(n->num_neurons_per_layer[layer_ptr], n->num_neurons_per_layer[layer_ptr - 1]);
    n->training->delta[layer_ptr] = (float64*)malloc(n->num_neurons_per_layer[layer_ptr] * sizeof(float64));
  }
  return n;
}

void save_network(Network *n, char *file) {
  fptr = fopen(file, "wb");
  if (!fptr) return;
  write_new_line(n->num_neurons_per_layer, sizeof(uint32), n->num_layers, fptr); /* architecture */

  uint32 num_weights = 0;
  for (uint32 i = 1; i < n->num_layers; i++) {
    num_weights += n->num_neurons_per_layer[i-1] * n->num_neurons_per_layer[i];
  }

  float64 *weights = (float64*)malloc(num_weights * sizeof(float64));
  uint32 idx = 0;
  for (uint32 i = 1; i < n->num_layers; i++) {
    for (uint32 j = 0; j < n->num_neurons_per_layer[i]; j++) {
      for (uint32 k = 0; k < n->layers[i]->neurons[j]->num_weights; k++) {
        weights[idx++] = n->layers[i]->neurons[j]->weights[k];
      }
    }
  }
  write_new_line(weights, sizeof(float64), num_weights, fptr); /* weights */

  free(weights);
  fclose(fptr);
}

Network *load_network(char *file) {
  Network *n;
  
  return n;
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
  The workflow is as follows:
  >> The input <<Neuron.value>> gets populated as the input.
  >> In the next layer, we choose the first <<Neuron>>. With each index of weight present on it,
  >> we take the i-th <<Neuron>> in the current layer, sum up the values taken from the <<Neuron.value>> 
  >> field in the current layer, and put the activation inside the <<Neuron.value>> field in the next layer.
  >> The same process is repeated for the second neuron in the second layer, and then this keeps happening
  >> until we get to the output layer, where a MSE/etc are calculated for the cost analysis.
*/

void forward_propagate(Network *network) {
  /* Enabling a 3-layer nested loop is **far** better for memory than constantly recalling the 
  forward propagation method.. */
  for (uint32 currLayerIdx = 0; currLayerIdx < network->num_layers - 1; currLayerIdx++) {
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
  }
}

void backward_propagate(Network *network, float64 *label_data) {
  /* compute the loss and do some magic stuff here to influence weights.. */
  uint32 output_layer = network->num_layers - 1;
  for (uint32 i = 0; i < network->num_neurons_per_layer[output_layer]; i++) {
    float64 gradient = network->training->derivative_function(network->layers[output_layer]->neurons[i]->value);
    network->training->delta[output_layer][i] = gradient * (network->layers[output_layer]->neurons[i]->value - label_data[i]);
    /* label data(could have a better name) is actually the desired output at each neuron in the output layer. */
  }

  for (uint32 currLayerIdx = network->num_layers - 2; currLayerIdx > 0; currLayerIdx--) {
    for (uint32 currNeuronIdx = 0; currNeuronIdx < network->num_neurons_per_layer[currLayerIdx]; currNeuronIdx++) {
      float64 sum = 0.0f;
      for (uint32 j = 0; j < network->num_neurons_per_layer[currLayerIdx + 1]; j++) {
        /* sum += weight of the neuron in the next layer corresponding to the neuron in the current layer * delta of each individual neuron in the next layer. */
        sum += network->layers[currLayerIdx + 1]->neurons[j]->weights[currNeuronIdx] * network->training->delta[currLayerIdx + 1][j];
        sum += network->layers[currLayerIdx + 1]->neurons[j]->bias * network->training->delta[currLayerIdx + 1][j];
      }

      float64 gradient = network->training->derivative_function(network->layers[currLayerIdx]->neurons[currNeuronIdx]->value);
      network->training->delta[currLayerIdx][currNeuronIdx] = gradient * sum;
    }
  }

  for (uint32 currLayerIdx = 0; currLayerIdx < network->num_layers - 1; currLayerIdx++) {
    for (uint32 i = 0; i < network->num_neurons_per_layer[currLayerIdx]; i++) {
      for (uint32 j = 0; j < network->num_neurons_per_layer[currLayerIdx + 1]; j++) {
        float64 dw = network->training->delta[currLayerIdx + 1][j] * network->layers[currLayerIdx]->neurons[i]->value;
        float64 db = network->training->delta[currLayerIdx + 1][j] * 1.0f;

        /* the weight of the neuron in the next layer corresponding to the neuron in the current layer = learning rate * delta */
        network->layers [currLayerIdx + 1]->neurons[j]->weights[i] -= network->training->learning_rate * dw;
        network->layers [currLayerIdx + 1]->neurons[j]->bias -= network->training->learning_rate * db;
      }
    }
  }
}

void train_network(Network *network, float64 **training_data, uint32 training_data_len, uint32 batch_len, float64 **label_data, uint32 label_data_len, uint32 epoch) {
  /* the training_data is a 2D array:
    - There are training_data_batch_len number of batches
    - Each batch has training_data_len number of inputs to be fed
  */
  uint32 i = 0;
  for (uint32 e = 0; e < epoch; e++) {
    for (i = 0; i < batch_len; i++) {
      float64 *retdata = test_network(training_data[i], training_data_len, network);
      backward_propagate(network, label_data[i]);
      
      if (e % 1000 == 0) {
        printf("\nEpoch: %d\nIn:", e);
        for (uint32 inlen = 0; inlen < training_data_len; inlen++) {
          printf("%f ", training_data[i][inlen]);
        }
        printf("\nOut: ");
        for (uint32 outlen = 0; outlen < label_data_len; outlen++) {
          printf("%f ", retdata[outlen]);
        }
        printf("\n====\n");
      }

      free(retdata); /* <== this is important! if we don't free it, we're looking at a LOT of memory leakage. */
    }
    // debug_network(network);
    // printf("\n\n===============\n\n");
  }
}

float64 *test_network(float64 *data, uint32 len, Network *network) {
  populate_input(data, len, network);
  forward_propagate(network);
  float64 *retdata = (float64*)malloc(network->num_neurons_per_layer[network->num_layers - 1] * sizeof(float64));
  for (uint32 i = 0; i < network->num_neurons_per_layer[network->num_layers - 1]; i++) {
    retdata[i] = network->layers[network->num_layers - 1]->neurons[i]->value;
  }
  network->currLayerIdx = 0;
  // debug_network(network);
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
With the backpropagation and training taken care of, the next step would be to solidify the 
permanence of the model - currently it all exists in the runtime, instead on the disk, so to load it
again on the next run we will need to save the architecture to file and then save the weights; so next time
we run the functions, we can easily transfer all of that to the functions.
*/