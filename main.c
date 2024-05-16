#include "nn.h"
#include <stdio.h>

int main() {
  /* set up the nodes */
  neuron *a = init_neuron(1.0f);
  set_input_neuron(a);
  set_output_neuron(a);

  f32 test_value = 119;
  f32 modifier = 17;
  train_nn(modifier);
  
  printf("Value: %f with a modifier %f; Expected output: %f, Actual output: %f\n", test_value, modifier, (f32)test_value/(f32)modifier, test_nn(test_value));
}