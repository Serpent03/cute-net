#include "nn.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>

/*
  New idea: instead of representing layers with structures
  how about representing with arrays? This way, CUDA integration can also 
  be done in a cleaner fashion!

  This can be represented with a matrix multiplication for simplicity.

  List    List      List
  Input1  Neuron1_1 Neuron2_1
  Input2  Neuron1_2 Neuron2_2
  Input3  Neuron1_3 Neuron2_3

*/



