#include "nn.h"
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
  uint32 layers[] = { 2, 2, 1 };
  uint32 network_wibr = sizeof(layers) / sizeof(layers[0]);

  uint32 input_neuron = layers[0];
  uint32 output_neuron = layers[network_wibr - 1];

  Network *nn = init_network(layers, network_wibr);


  float64 **training_data = (float64**)malloc(4 * sizeof(float64*));
  float64 **label_data = (float64**)malloc(4 * sizeof(float64*));

  uint32 ptr = 0;
  float64 t1[2] = {0, 0};
  insert_into_2d_array(training_data, t1, 2, &ptr);
  float64 t2[2] = {0, 1};
  insert_into_2d_array(training_data, t2, 2, &ptr);
  float64 t3[2] = {1, 0};
  insert_into_2d_array(training_data, t3, 2, &ptr);
  float64 t4[2] = {1, 1};
  insert_into_2d_array(training_data, t4, 2, &ptr);

  ptr = 0;
  float64 t5[1] = {0};
  insert_into_2d_array(label_data, t5, 1, &ptr);
  float64 t6[1] = {1};
  insert_into_2d_array(label_data, t6, 1, &ptr);
  float64 t7[1] = {1};
  insert_into_2d_array(label_data, t7, 1, &ptr);
  float64 t8[1] = {0};
  insert_into_2d_array(label_data, t8, 1, &ptr);

  train_network(nn, training_data, 2, 4, label_data, 2, 5000);


  float64 data[2] = { 0, 1 };
  float64 *res = test_network(data, input_neuron, nn);
  for (uint32 i = 0; i < output_neuron; i++) {
    printf("%f ", res[i]);
  }
  printf("\n");
  free(res);

  data[0] = 1;
  data[1] = 1;
  res = test_network(data, input_neuron, nn);
  for (uint32 i = 0; i < output_neuron; i++) {
    printf("%f ", res[i]);
  }
  printf("\n");
  free(res);
}