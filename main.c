#include "nn.h"
#include <stdio.h>

int main() {
  /* set up the nodes */
  uint32 layers[] = { 2, 2, 2 };
  uint32 network_wibr = sizeof(layers) / sizeof(layers[0]);
  uint32 input_neuron = layers[0];
  uint32 output_neuron = layers[network_wibr - 1];

  Network *nn = init_network(layers, network_wibr);
  
  // populate_input(data, 2, nn);
  // debug_network(nn);

  // forward_propagate(nn);
  // debug_network(nn);

  // forward_propagate(nn);
  // debug_network(nn);

  float64 data[2] = { 0.05, 0.10 };
  float64 *res = test_network(data, input_neuron, nn);
  for (uint32 i = 0; i < output_neuron; i++) {
    printf("%f ", res[i]);
  }
  free(res);
}