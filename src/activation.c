#include "../include/nn.h"
#include <math.h>

float64 leakyRELU(float64 value) {
  /* ReLU is pretty much max(0, value). if we get negative values, however, we can quickly get
  stuck. In that scenario, we simply do max(value * -0.001, value). If the value is positive, then we
  choose it, otherwise we reduce its magnitude and change the direction, and take it. */
  float64 activation = value > value * 0.01 ? value : value * 0.01;
  return activation;
}

float64 leakyRELU_d(float64 value) {
  float64 derv = (value > 0) ? 1.0f : 0.5f;
  return derv;
}

float64 sigmoid(float64 value) {
  /* Sigmoid introduces non-linearity by compressing values close to -1..1 to their original
  magnitude, and squishes down the values of numbers greater than that by exponentially clamping down
  such that [-infinity, +infinity] becomes [-1, +1]. */
  return (1 / (1 + powl(2.71828182846, -value)));
}

float64 sigmoid_d(float64 value) {
  return value * (1 - value);
}

float64 meanSqErr(float64 output, float64 input_label) {
  return 0.5f * (float64)pow(output - input_label, 2);
}