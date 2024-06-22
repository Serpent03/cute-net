#include "nn.h"
#include <stdio.h>

int main() {
  /* set up the nodes */
  uint32 layers[3] = { 2, 2, 1 };
  Network *nn = init_network(layers, 3);
  debug_network(nn);
}