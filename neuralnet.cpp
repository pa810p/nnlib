/*
 * File neuralnet.cpp
 */

#include "neuralnet.h"

#include <cmath>
#include <stdexcept>
#include <vector>

NeuralNet::NeuralNet(std::span<const int> layer_sizes, int inputs,
                     double min_weights, double max_weights, double eta) {
	if (layer_sizes.empty() || inputs <= 0)
		throw std::invalid_argument("NeuralNet requires positive layer and input counts");

	for (const int layer_size : layer_sizes)
		if (layer_size <= 0)
			throw std::invalid_argument("NeuralNet layer sizes must be positive");

	input_size = static_cast<std::size_t>(inputs);
	layers.resize(layer_sizes.size());

	for (std::size_t i = 0; i < layer_sizes.size(); ++i) {
		const int layer_inputs = (i == 0) ? inputs : layer_sizes[i - 1];
		const int activation = (i == 0) ? ACT_SIGMOID :
			(i + 1 == layer_sizes.size() ? ACT_LINEAR : ACT_SIGMOID);

		if (layers[i].Initialize(layer_sizes[i], layer_inputs, min_weights,
		                         max_weights, activation, 1, eta) != 0)
			throw std::runtime_error("Failed to initialize neural network layer");
	}

	last_layer_size = static_cast<std::size_t>(layers.back().getLayerSize());
}

NeuralNet::NeuralNet(std::span<const int> layer_sizes, int inputs)
	: NeuralNet(layer_sizes, inputs, 0.0, 1.0, 0.5) {}

void NeuralNet::Check(std::span<const double> inputs, std::span<double> outputs) {
	if (inputs.size() != input_size)
		throw std::invalid_argument("NeuralNet input size does not match initialized input count");
	if (outputs.size() != last_layer_size)
		throw std::invalid_argument("NeuralNet output size does not match output layer size");

	layers.front().Train(inputs);
	for (std::size_t i = 1; i < layers.size(); ++i)
		layers[i].Train(layers[i - 1]);

	for (std::size_t i = 0; i < last_layer_size; ++i)
		outputs[i] = layers.back().getOutput(static_cast<int>(i));
}

double NeuralNet::TrainEpoch(int sample_count, std::span<const double> inputs,
                             std::span<const double> outputs) {
	double sum = 0.0;
	std::vector<double> error(last_layer_size);

	for (int sample = 0; sample < sample_count; ++sample) {
		const auto input_offset = static_cast<std::size_t>(sample) * input_size;
		const auto output_offset = static_cast<std::size_t>(sample) * last_layer_size;
		layers.front().Train(inputs.subspan(input_offset, input_size));

		for (std::size_t i = 1; i < layers.size(); ++i)
			layers[i].Train(layers[i - 1]);

		layers.back().countLastDelta(
			outputs.subspan(output_offset, last_layer_size), error);

		for (std::size_t i = layers.size() - 1; i > 0; --i)
			layers[i - 1].countDelta(layers[i]);

		for (NeuralLayer& layer : layers)
			layer.Modify();

		for (const double value : error)
			sum += value * value;
	}

	return std::sqrt(sum / (static_cast<double>(sample_count) * last_layer_size));
}

void NeuralNet::Train(int sample_count, std::span<const double> inputs,
                      std::span<const double> outputs, double error, long epochs) {
	if (sample_count <= 0)
		throw std::invalid_argument("Training sample count must be positive");

	const auto samples = static_cast<std::size_t>(sample_count);
	if (inputs.size() % input_size != 0 || inputs.size() / input_size != samples)
		throw std::invalid_argument("Training input size does not match sample count and network input size");
	if (outputs.size() % last_layer_size != 0 || outputs.size() / last_layer_size != samples)
		throw std::invalid_argument("Training output size does not match sample count and network output size");

	double current_error = TrainEpoch(sample_count, inputs, outputs);
	for (long epoch = 0; current_error > error && epoch < epochs; ++epoch)
		current_error = TrainEpoch(sample_count, inputs, outputs);
}

int NeuralNet::setActivation(int layer, int activation) {
	if (layer < 0 || static_cast<std::size_t>(layer) >= layers.size())
		return -1;
	return layers[static_cast<std::size_t>(layer)].setActivation(activation);
}

int NeuralNet::setActivation(int layer, double (*activation)(double),
                             double (*derivative)(double)) {
	if (layer < 0 || static_cast<std::size_t>(layer) >= layers.size())
		return -1;
	return layers[static_cast<std::size_t>(layer)].setActivation(activation, derivative);
}

void NeuralNet::setAllActivation(int activation) {
	for (NeuralLayer& layer : layers)
		layer.setActivation(activation);
}

void NeuralNet::setAllActivation(double (*activation)(double),
                                 double (*derivative)(double)) {
	for (NeuralLayer& layer : layers)
		layer.setActivation(activation, derivative);
}

void NeuralNet::setEta(double eta) {
	for (NeuralLayer& layer : layers)
		layer.setEta(eta);
}
