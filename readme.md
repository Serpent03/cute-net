
# Cute-Net
Cute-Net is a simple neural network made in C that is able to perform deep neural network learning using generalized functions.

## Pre-Requisites
- C compiler (gcc/minGW)
- Bash(if you are on windows, add the Github Bash to PATH)
- Make

## Installation

Clone the repository onto your local machine using git:

```bash
git clone https://github.com/Serpent03/cute-net.git
```

Before starting, build the project using `make`, and run the project using `make run`. The `./main` executable is generated inside `src/`.

## Overview

There are two files of main interest:
- `nn.h`: All the headers, definitions and structures are stored in this file.
- `nn.c`: All functions, neural network logic and other relevant material are defined here.

## How does it work?

CuteNet is a neural network able to generate network in a generalized fashion - any network from 1-1-1 to complex, MNIST-able networks like 784-16-16-10 can be generated using the generalized function.

In `main.c`, there are some functions of interest:
- `init_network`: This initializes a neural network of given dimensions.
- `train_network`: This trains the network according to the given input.
- `test_network`: This tests the network according to the current network weights.
- `save_network` and `load_network` save and load the network to disk.

A general approach to using the functions would be:
```C
int layers[] = { 2, 5, 3, 1 };
int num_layers = 4;
Network *nn = init_network(layers, num_layers);

double **training_data, **label_data; // assume these are already populated.
int num_input_layer_neurons = 2;
int num_output_layer_neurons = 1;
int batches = 4;
int epoch = 100;
float learning_rate = 0.1; /* values around this range work well for a sigmoid function. */
train_network(nn, training_data, num_input_layer_neurons, num_batches, label_data, num_output_layer_neurons, epoch, learning_rate);
```

`layers` defines the number of neurons at each layer in the network. `layers[0]` is the input layer, and `layers[num_layers - 1]` is the output layer. Everything in between are the hidden layers.

To save and load the model from the disk, the following commands can be used:
```C
Network *nn; /* assume it is already trained */
save_network(nn, "./network/network.net");

Network *copy_nn = load_network("./network/network.net");
/* further operations on <<copy_nn>> can be done completely independent of <<nn>>. */
```

## Future TODOs
- Evolve from perceptron to neural network ✅
- Generalize neural network operations ✅
- Save/Load model from disk ✅
- Better I/O utility for large datasets.
- Integration with CUDA.
- Ability to read images(MNIST!), and other media to detect features.
