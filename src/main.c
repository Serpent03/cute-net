#include "../include/nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void insert_into_2d_array(float64 **in_array, float64 *append_array, uint32 a, uint32 *l) {
  in_array[*l] = (float64*)calloc(a, sizeof(float64));
  memcpy(in_array[*l], append_array, a * sizeof(float64));
  (*l)++;
}

int main() {
  /* set up the nodes */
  uint32 layers[] = { 2, 5, 4, 1 };
  uint32 network_wibr = sizeof(layers) / sizeof(layers[0]);

  uint32 input_neuron = layers[0];
  uint32 output_neuron = layers[network_wibr - 1];

  Network *nn = init_network(layers, network_wibr, ACTIVATION_RELU, 0.05);

  /** @todo fix thsi wacky ass input methodology. */

  float64 **training_data = (float64**)malloc(4 * sizeof(float64*));
  float64 **label_data = (float64**)malloc(4 * sizeof(float64*));

  uint32 ptr = 0;
  float64 t1[] = {1, 1};
  uint32 layer_size = 2;
  insert_into_2d_array(training_data, t1, layer_size, &ptr);
  float64 t2[] = {0, 1};
  insert_into_2d_array(training_data, t2, layer_size, &ptr);
  float64 t3[] = {1, 0};
  insert_into_2d_array(training_data, t3, layer_size, &ptr);
  float64 t4[] = {1, 1};
  insert_into_2d_array(training_data, t4, layer_size, &ptr);

  ptr = 0;
  float64 t5[1] = {0};
  uint32 layer_size_2 = 1;
  insert_into_2d_array(label_data, t5, layer_size_2, &ptr);
  float64 t6[1] = {1};
  insert_into_2d_array(label_data, t6, layer_size_2, &ptr);
  float64 t7[1] = {1};
  insert_into_2d_array(label_data, t7, layer_size_2, &ptr);
  float64 t8[1] = {0};
  insert_into_2d_array(label_data, t8, layer_size_2, &ptr);

  /* what about seperating a directory for both training and testing set?
    that way, I only have to feed in the data into the commandline arguments,
    along with other hyperparameters like epoch, learning rate, and whether to save the file or not.
   */


  train_network(nn, training_data, layer_size, ptr, label_data, layer_size_2, 10000);
  save_network(nn, "./network/network.net");

  Network *copy = load_network("./network/network.net");

  float64 data[] = { 1, 1 };
  float64 *res = test_network(data, input_neuron, copy);
  for (uint32 i = 0; i < output_neuron; i++) {
    printf("%f ", res[i]);
  }
  printf("\n");
  free(res);

  data[0] = 0;
  data[1] = 1;
  res = test_network(data, input_neuron, copy);
  for (uint32 i = 0; i < output_neuron; i++) {
    printf("%f ", res[i]);
  }
  printf("\n");
  free(res);

}
