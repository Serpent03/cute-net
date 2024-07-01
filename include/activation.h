#include "common.h"

/**
 * @brief Leaky ReLU activation function.
 * @param value The non-activated value in question.
 * @return activated float64 value.
*/
float64 leakyRELU(float64 value);

/**
 * @brief Leaky RELU derivative function.
 * @param value The activated value in question.
 * @return derivative as a float64.
*/
float64 leakyRELU_d(float64 value);

/**
 * @brief Sigmoid activation function.
 * @param value The non-activated value in question.
 * @return activated float64 value.
*/
float64 sigmoid(float64 value);

/**
 * @brief Sigmoid derivative function.
 * @param value The activated value in question.
 * @return derivative as a float64.
*/
float64 sigmoid_d(float64 value);

/**
 * @brief Mean Square Error cost function.
 * @param output Output at the end of the network.
 * @param input_label The expected output fed with the training data.
 * @return mean squared error as a float64
*/
float64 meanSqErr(float64 output, float64 input_label); 