#pragma once

/* ==== DEFINITIONS & TYPES ==== */

#define IN_NODES 2
#define OUT_NODES 1

typedef unsigned char u8;
typedef char i8;

typedef unsigned short int u16;
typedef short int i16;

typedef unsigned int u32;
typedef int i32;

typedef float f32;
typedef double f64;

/* ==== STRUCTURES & METHODS ==== */

typedef struct neuron {
  f32 data; /* data stored inside the neuron. initially 1.0f */
  f32 weight; /* weight from behind */
  struct neuron *next; /* next neuron */
} neuron;

neuron *input_neuron;
neuron *output_neuron;

/**
 * @brief Initialize a neuron and return a pointer to the structure.
 * @param weight Weight attached to the neuron.
 * @return A pointer to a neuron structure.
*/
neuron* init_neuron(f32 weight);

/**
 * @brief Attach a neuron to be next in chain.
 * @param n The neuron to attach to.
 * @param weight The weight for the new neuron.
 * @return A pointer to the new neuron attached.
*/
neuron* attach_neuron(neuron *n, f32 weight);

/**
 * @brief Forward the activation from the current neuron to the next one.
 * @param n The neuron to propagate the data from.
*/
void propagate(neuron *n);

/**
 * @brief Cycle the data through the first hidden layer.
 * @param data A 4-byte float.
*/
void set_input(f32 data);

/**
 * @brief Get the output from the output node.
 * @return A 4-byte float.
*/
f32 get_output();

/**
 * @brief Set the input neuron layer.
 * @param n The input neuron.
*/
void set_input_neuron(neuron *n);

/**
 * @brief Set the output neuron layer.
 * @param n The output neuron.
*/
void set_output_neuron(neuron *n);

/**
 * @brief Train the neural network on a specific modifier. The network will try
 * to get the activation of the nodes as close as possible to the modifier.
 * @param modifier Configuration value to set the neural network target towards a mathematical factor.
*/
void train_nn(f32 modifier);

/**
 * @brief Test the neural network against a specific value.
 * @param input The input parameter.
 * @return The resulting output.
*/
f32 test_nn(f32 input);