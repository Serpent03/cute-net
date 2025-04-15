#include "../include/nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

float64 **convert_2d_stack_array_to_heap(float64 *arr, uint32 row, uint32 col) {
  float64 **a = (float64**)malloc(row * sizeof(float64*));
  for (uint32 r = 0; r < row; r++) {
    a[r] = (float64*)malloc(col * sizeof(float64));
    memcpy(a[r], &arr[r * col], col * sizeof(float64));
  }
  return a;
}

void free_2d_heap_array(float64 **a, uint32 row, uint32 col) {
  for (uint32 r = 0; r < row; r++) {
    if (a[r] != NULL) free(a[r]);
  }
  if (a != NULL) free(a);

  return;
}

int main() {
  /* set up the nodes */
  uint32 layers[] = { 2, 2, 3, 1 };
  uint32 network_wibr = sizeof(layers) / sizeof(layers[0]);

  uint32 input_neuron = layers[0];
  uint32 output_neuron = layers[network_wibr - 1];

  Network *nn = init_network(layers, network_wibr, ACTIVATION_RELU, 0.05);

  /* convert this stack-based array to a heap() allocated array,
   * since the <<train_network>> function expects a heap-allocated
   * arrray. */
  float64 training_data[4][2] = {
    {0, 1},
    {1, 0},
    {0, 0},
    {1, 1}
  };
  float64 label_data[4][1] = {
    {1},
    {1},
    {0},
    {0}
  };

  float64 **training_data_heap = convert_2d_stack_array_to_heap(&training_data[0][0], 4, 2);
  float64 **testing_data_heap = convert_2d_stack_array_to_heap(&label_data[0][0], 4, 1);

  uint32 input_layer_size = 2;
  uint32 output_layer_size = 1;
  uint32 batches = 4;
  train_network(
      nn, 
      training_data_heap, 
      input_layer_size, 
      batches, 
      testing_data_heap, 
      output_layer_size, 
      10000
  );
  save_network(nn, "./network/network.net");
  free_network(nn);

  Network *copy = load_network("./network/network.net");

  float64 data[] = { 1, 1 };
  float64 *res = test_network(data, input_neuron, copy);
  printf("EXPECTED: 0\nACTUAL: ");
  for (uint32 i = 0; i < output_neuron; i++) {
    printf("%f ", res[i]);
  }
  printf("\n");
  free(res);

  data[0] = 0;
  data[1] = 1;
  printf("\nEXPECTED: 1\nACTUAL: ");
  res = test_network(data, input_neuron, copy);
  for (uint32 i = 0; i < output_neuron; i++) {
    printf("%f ", res[i]);
  }
  printf("\n");
  free(res);

}
