
# Cute-Net
Cute-Net is a simple neural network made in C that can try to learn linear mathematical relationships.

## Pre-Requisites
- C compiler (gcc/minGW)
- Bash(if you are on windows, add the Github Bash to PATH)
- Make

## Installation

Clone the repository onto your local machine using git:

```bash
git clone https://github.com/Serpent03/cute-net.git
```

Before starting, build the project using `make`, and run the project using `./main`.

## Overview

There are two files of main interest:
- `nn.h`: All the headers, definitions and structures are stored in this file.
- `nn.c`: All functions, neural network logic and other relevant material are defined here.

## How does it work?

Cute-net is a very tiny neural network: it only consists of one neural node(in other words, a perceptron), that tries to fit the given mathematical relation using gradient descent to the find the least error in the cost function. 

In `main.c`, there are some functions of interest:
- `init_neuron`: This initializes a neuron node.
- `set_input_neuron`: This registers the neuron as the first hidden layer.
- `set_output_neuron`: This registers the neuron as the last hidden layer.
- `train_nn`: This actually "trains" the weight attached to the neuron node.
- `test_nn`: This tests the given input.   

## Future TODOs
- Add multi-layer, multi-dimensional neural nodes for a far more complex and capable system.
- Ability to read images, and other media to detect features.