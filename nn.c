#include "nn.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>

neuron *init_neuron(f32 weight) {
  neuron *n = (neuron*)calloc(1, sizeof(neuron));
  n->data = 1;
  n->weight = weight;
  n->next = NULL;
  return n;
}

neuron *attach_neuron(neuron *n, f32 weight) {
  n->next = (neuron*)calloc(1, sizeof(neuron));
  n->next->data = 1;
  n->next->weight = weight;
  n->next->next = NULL;
  return n->next;
}

void propagate(neuron *n) {
  assert(n->next); /* verify if the next neuron even exists. */
  n->next->data = n->data * n->next->weight;
}

void set_input(f32 data) {
  assert(input_neuron); /* Make sure the input neuron exists. */
  input_neuron->data = input_neuron->weight * data;
}

f32 get_output() {
  assert(output_neuron); /* Make sure the output neuron exists. */
  return output_neuron->data;
}

void set_input_neuron(neuron *n) {
  input_neuron = n;
}

void set_output_neuron(neuron *n) {
  output_neuron = n;
}

void train_nn(f32 modifier) {
  f32 expected, actual, error, last_error;
  error = 0;
  last_error = 0;

  i8 direction = 1;
  u32 batch = 50;
  u32 _b = batch;

  printf("Training the neural network with a modifier of: %f and a batch of %d\n\n", modifier, batch);

  for (i32 i = -20; i <= 20; i++) {
    /* The general idea should now be to minimize the error
    for each input. This should be achieved by keeping a while loop inside this for loop.
    For every set of input, we run the while loop to minimize the error until we've reached
    the local minima. */

    _b = batch;

    while (_b--) {
      set_input(i);

      expected = (f32)i / modifier;
      actual = get_output();

      last_error = error;
      error = pow(expected - actual, 2);


      if (error > last_error) {
        /* Go the other way, if the error has been increasing.
        Ideally, the neuron would be modified to achieve minimum cost through
        adjusting the factor by how close it is to the minimum(through change
        of slope). Using 0.001f as a fixed value here is a very simple P-controller 
        based approach. */

        direction = direction * -1;
        input_neuron->weight += 0.001f * direction;
      } else {
        input_neuron->weight += 0.001f * direction;
      } 
    }
    printf("Expected: %f Actual: %f, E: %f, w: %f\n", expected, actual, error, input_neuron->weight);
  }
  printf("\n\n");
}

f32 test_nn(f32 input) {
  set_input(input);
  return get_output();
}
