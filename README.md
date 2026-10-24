# nnlib

A small C++ neural network library.

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

The project uses CMake 3.31 or newer and requires C++23 support. Glaze 9.0.0
documents tested compiler versions of GCC 13+, Clang 18+, and MSVC 14.50+.
Check the installed CMake version with `cmake --version`; install a newer
version from [cmake.org/download](https://cmake.org/download/) if needed. The
build fetches the pinned Glaze release through CMake FetchContent. Configure
and build the demonstration executable, shared library, static library, and
tests with:

```sh
cmake -S . -B build
cmake --build build
```

The library's input and output buffer APIs use `std::span` and reject spans
whose sizes do not match the configured network dimensions. Neural network
models are persisted as versioned JSON files; custom activations must be
registered with a stable identifier before they can be saved or loaded.

Save a trained model with `network.save("model.json")` and restore it with
`NeuralNet::load("model.json")`. The JSON stores topology, activation IDs,
weights, biases, learning rates, momentum coefficients, and momentum buffers.
Register custom activation functions with `registerActivation` and select them
by their `custom.*` ID before saving; register the same ID in the loading
application before loading the model. Function pointers are never written to
the model file.

Run the demonstration using the included training data:

```sh
./build/demo < tests/resources/trains.json
```

The datasets are available as JSON files in `tests/resources/`.

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
`build-coverage/coverage_html/index.html` for the HTML report. Coverage is
filtered to the library's `.cpp` implementation files and saved to
`build-coverage/coverage_filtered.info`; the unfiltered capture is in
`build-coverage/coverage.info`.

Equivalent manual commands are:

```sh
cmake -S . -B build-coverage -DCMAKE_BUILD_TYPE=Debug -DNNLIB_ENABLE_COVERAGE=ON
cmake --build build-coverage --target coverage
```

## Source code

The implementation is organized into a few core components:

| Files | Purpose |
|---|---|
| `include/nnl/neuron.h`, `src/neuron.cpp` | Neuron calculations and connection weights |
| `include/nnl/layer.h`, `src/layer.cpp` | Layers of neurons |
| `include/nnl/activation.h`, `src/activation.cpp` | Activation functions |
| `include/nnl/neuralnet.h`, `src/neuralnet.cpp` | Network construction, propagation and training |
| `app/main.cpp` | Demonstration program |
| `tests/test_activation.cpp` | Activation function tests |
| `tests/test_neuron.cpp` | Neuron unit tests |
| `tests/test_layer.cpp` | Layer unit tests |
| `tests/test_neuralnet.cpp` | Network API tests |
| `tests/test_neuralnet_scenarios.cpp` | End-to-end network training scenarios |
| `tests/test_persistence.cpp` | Model round-trip and validation tests |

The code is small enough to follow from the individual neuron calculations through layer processing to network training.

## History

The [`nnlib-1.0` tag](https://github.com/pa810p/nnlib/tree/nnlib-1.0) marks the original version of the project, written around 2001. It provides a fixed reference point for the code as it was when the project was first published on GitHub.

Future development may focus on refactoring the implementation using modern C++ practices, including RAII and safer resource management, while preserving the original algorithms and educational purpose.

The original version remains available in the repository history for reference.

## License

This project is distributed under the GNU General Public License, version 2.0. See [LICENSE](LICENSE) for details.
