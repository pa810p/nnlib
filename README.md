# nnlib

A small C++ neural network library, originally written around 2001.

## About

`nnlib` is an implementation of feed-forward neural networks with hidden layers, trained using the backpropagation algorithm.

I wrote the original code around 2001 as an exploration of how neural networks work internally: how neurons process inputs, how signals propagate through layers, and how connection weights are adjusted during training.

The code is deliberately straightforward, with the core neural-network operations implemented directly in C++. The intention is to make the underlying algorithms accessible to anyone interested in understanding how a neural network learns, rather than hiding the implementation behind a high-level framework.

## Features

- **Feed-forward neural networks** with configurable layer sizes and hidden layers.
- **Backpropagation** for adjusting connection weights during training.
- **Configurable activation functions**, including sigmoid and linear activation.
- **Configurable learning parameters**, including the learning rate.
- **Training with supplied examples**, allowing the network to learn from input/output pairs.
- **C++ implementation** without a dependency on an external machine-learning framework.

The default configuration uses sigmoid activation in hidden layers and linear activation in the output layer.

## Building

The project includes a Makefile and requires GNU Make and a C++ compiler compatible with the source code.

Build the project:

```sh
make
```

The build produces the demonstration executable `demo`, a shared library `libnn.so`, and a static library `libnn.a`.

Run the demonstration using the included training data:

```sh
./demo < trains
```

An additional dataset, `trains2`, is also included.

Run the test suite:

```sh
make test
```

Remove generated build artifacts:

```sh
make clean
```

## Source code

The implementation is organized into a few core components:

| Files | Purpose |
|---|---|
| `neuron.h`, `neuron.cpp` | Neuron calculations and connection weights |
| `layer.h`, `layer.cpp` | Layers of neurons |
| `activation.h`, `activation.cpp` | Activation functions |
| `neuralnet.h`, `neuralnet.cpp` | Network construction, propagation and training |
| `main.cpp` | Demonstration program |
| `tests/` | Tests for the implementation |

The code is small enough to follow from the individual neuron calculations through layer processing to network training.

## History

The [`nnlib-1.0` tag](https://github.com/pa810p/nnlib/tree/nnlib-1.0) marks the original version of the project, written around 2001. It provides a fixed reference point for the code as it was when the project was first published on GitHub.

Future development may focus on refactoring the implementation using modern C++ practices, including RAII and safer resource management, while preserving the original algorithms and educational purpose.

The original version remains available in the repository history for reference.

## License

This project is distributed under the GNU General Public License, version 2.0. See [LICENSE](LICENSE) for details.
