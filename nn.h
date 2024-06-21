#pragma once

/* ==== DEFINITIONS & TYPES ==== */

#define IN_NODES 2
#define OUT_NODES 1

typedef unsigned char uint8;
typedef char int8;

typedef unsigned short int uint16;
typedef short int int16;

typedef unsigned int uint32;
typedef int int32;

typedef float float32;
typedef double float64;

/* ==== STRUCTURES & METHODS ==== */

typedef struct {
  float64 weight;
  float64 bias;
  float64 result;
} Neuron;