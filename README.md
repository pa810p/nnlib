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

The project uses CMake 3.20 or newer and a C++20 compiler. Configure and build
the demonstration executable, shared library, and static library with:

```sh
cmake -S . -B build
cmake --build build
```

Run the demonstration using the included training data:

```sh
./build/demo < trains
```

An additional dataset, `trains2`, is also included.

## Testing

Tests use GoogleTest. Install its development package before configuring the
project. On Debian or Ubuntu:

```sh
sudo apt install libgtest-dev
```

Configure and build the project, then run the tests with CTest:

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## Coverage

Coverage is collected for the neural-network library code, not the test runner
or demo application. It requires GCC, `gcov`, `lcov`, and `genhtml`. On Debian
or Ubuntu, install the coverage tools with:

```sh
sudo apt install lcov
```

From the project root, run:

```sh
./build_coverage.sh
```

The script configures a separate instrumented build, runs the tests, and
generates the report. Open
`build-coverage/coverage_html/index.html` for the HTML report; raw lcov data is
saved to `build-coverage/coverage.info`.

Equivalent manual commands are:

```sh
cmake -S . -B build-coverage -DCMAKE_BUILD_TYPE=Debug -DNNLIB_ENABLE_COVERAGE=ON
cmake --build build-coverage --target coverage
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
| `tests/test_activation.cpp` | Activation function tests |
| `tests/test_neuron.cpp` | Neuron unit tests |
| `tests/test_layer.cpp` | Layer unit tests |
| `tests/test_neuralnet.cpp` | Network API tests |
| `tests/test_neuralnet_scenarios.cpp` | End-to-end network training scenarios |

The code is small enough to follow from the individual neuron calculations through layer processing to network training.

## History

The [`nnlib-1.0` tag](https://github.com/pa810p/nnlib/tree/nnlib-1.0) marks the original version of the project, written around 2001. It provides a fixed reference point for the code as it was when the project was first published on GitHub.

Future development may focus on refactoring the implementation using modern C++ practices, including RAII and safer resource management, while preserving the original algorithms and educational purpose.

The original version remains available in the repository history for reference.

## License

This project is distributed under the GNU General Public License, version 2.0. See [LICENSE](LICENSE) for details.
