/*
 * File neuralnet.cpp
 */

#include <nnl/neuralnet.h>

#include <glaze/json.hpp>

#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr unsigned int model_format_version = 1;

struct NeuronModel {
	std::string activation;
	std::vector<double> weights;
	double bias_input = 0.0;
	double bias_weight = 0.0;
	double learning_rate = 0.0;
	double momentum_coefficient = 0.0;
	std::vector<double> momentum;
	double bias_momentum = 0.0;
};

struct LayerModel {
	std::size_t input_size = 0;
	std::vector<NeuronModel> neurons;
};

struct NetworkModel {
	std::string format;
	unsigned int version = 0;
	std::size_t input_size = 0;
	std::vector<LayerModel> layers;
};

constexpr glz::opts model_read_options{.error_on_missing_keys = true};

bool all_finite(const std::vector<double>& values) {
	for (const double value : values)
		if (!std::isfinite(value))
			return false;
	return true;
}

void validate_model(const NetworkModel& model) {
	if (model.format != "nnlib-model")
		throw std::runtime_error("Unsupported model format");
	if (model.version != model_format_version)
		throw std::runtime_error("Unsupported model version: " + std::to_string(model.version));
	if (model.input_size == 0 || model.input_size > static_cast<std::size_t>(std::numeric_limits<int>::max()))
		throw std::runtime_error("Invalid model input size");
	if (model.layers.empty())
		throw std::runtime_error("Model must contain at least one layer");

	std::size_t expected_input_size = model.input_size;
	for (const LayerModel& layer : model.layers) {
		if (layer.input_size != expected_input_size || layer.neurons.empty() ||
		    layer.neurons.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
			throw std::runtime_error("Invalid model layer dimensions");

		for (const NeuronModel& neuron : layer.neurons) {
			if (!resolveActivation(neuron.activation))
				throw std::runtime_error("Unknown activation ID: " + neuron.activation);
			if (neuron.weights.size() != expected_input_size ||
			    neuron.momentum.size() != expected_input_size)
				throw std::runtime_error("Invalid model neuron dimensions");
			if (!std::isfinite(neuron.bias_input) || neuron.bias_input < 0.0 ||
			    !std::isfinite(neuron.bias_weight) ||
			    !std::isfinite(neuron.learning_rate) ||
			    !std::isfinite(neuron.momentum_coefficient) ||
			    !std::isfinite(neuron.bias_momentum) ||
			    !all_finite(neuron.weights) || !all_finite(neuron.momentum))
				throw std::runtime_error("Model contains a non-finite or invalid neuron value");
			if (neuron.bias_input == 0.0 &&
			    (neuron.bias_weight != 0.0 || neuron.bias_momentum != 0.0))
				throw std::runtime_error("Model contains bias parameters without a bias input");
		}
		expected_input_size = layer.neurons.size();
	}
}

}

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
	const TrainingOptions options{
		.target_rmse = error,
		.max_epochs = epochs
	};
	Train(sample_count, inputs, outputs, options);
}

void NeuralNet::Train(int sample_count, std::span<const double> inputs,
                      std::span<const double> outputs, const TrainingOptions& options) {
	if (sample_count <= 0)
		throw std::invalid_argument("Training sample count must be positive");

	const auto samples = static_cast<std::size_t>(sample_count);
	if (inputs.size() % input_size != 0 || inputs.size() / input_size != samples)
		throw std::invalid_argument("Training input size does not match sample count and network input size");
	if (outputs.size() % last_layer_size != 0 || outputs.size() / last_layer_size != samples)
		throw std::invalid_argument("Training output size does not match sample count and network output size");

	double current_error = TrainEpoch(sample_count, inputs, outputs);
	std::size_t completed_epochs = 1;
	if (options.on_epoch)
		options.on_epoch(completed_epochs, current_error);
	for (long epoch = 0; current_error > options.target_rmse &&
	                     epoch < options.max_epochs; ++epoch) {
		current_error = TrainEpoch(sample_count, inputs, outputs);
		++completed_epochs;
		if (options.on_epoch)
			options.on_epoch(completed_epochs, current_error);
	}
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

int NeuralNet::setActivation(int layer, std::string_view activation) {
	if (layer < 0 || static_cast<std::size_t>(layer) >= layers.size())
		return -1;
	return layers[static_cast<std::size_t>(layer)].setActivation(activation);
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

void NeuralNet::setMomentum(double momentum) {
	for (NeuralLayer& layer : layers)
		layer.setMomentum(momentum);
}

void NeuralNet::save(const std::filesystem::path& path) const {
	NetworkModel model;
	model.format = "nnlib-model";
	model.version = model_format_version;
	model.input_size = input_size;
	model.layers.reserve(layers.size());

	for (const NeuralLayer& layer : layers) {
		LayerModel layer_model;
		layer_model.input_size = static_cast<std::size_t>(layer.input_size);
		layer_model.neurons.reserve(static_cast<std::size_t>(layer.layer_size));

		for (int i = 0; i < layer.layer_size; ++i) {
			const Neuron& neuron = layer.neurons[i];
			if (neuron.activation_id.empty())
				throw std::runtime_error("Cannot save a network with an unregistered activation");
			if (layer_model.neurons.empty() && !resolveActivation(neuron.activation_id))
				throw std::runtime_error("Unknown activation ID: " + neuron.activation_id);

			NeuronModel neuron_model;
			neuron_model.activation = neuron.activation_id;
			neuron_model.weights.assign(neuron.w.begin(), neuron.w.begin() + neuron.inputs);
			neuron_model.bias_input = neuron.constant;
			if (neuron.constant > 0.0) {
				neuron_model.bias_weight = neuron.w[static_cast<std::size_t>(neuron.inputs)];
				neuron_model.bias_momentum = neuron.momentum[static_cast<std::size_t>(neuron.inputs)];
			}
			neuron_model.learning_rate = neuron.ni;
			neuron_model.momentum_coefficient = neuron.inertia;
			neuron_model.momentum.assign(neuron.momentum.begin(),
			                             neuron.momentum.begin() + neuron.inputs);
			layer_model.neurons.push_back(std::move(neuron_model));
		}
		model.layers.push_back(std::move(layer_model));
	}
	validate_model(model);

	std::string json;
	const auto write_error = glz::write<glz::opts{.prettify = true}>(model, json);
	if (write_error)
		throw std::runtime_error("Failed to serialize network model: " +
		                         glz::format_error(write_error, json));

	std::ofstream file(path, std::ios::binary | std::ios::trunc);
	if (!file)
		throw std::runtime_error("Unable to open model file for writing: " + path.string());
	file.write(json.data(), static_cast<std::streamsize>(json.size()));
	if (!file)
		throw std::runtime_error("Failed to write model file: " + path.string());
}

NeuralNet NeuralNet::load(const std::filesystem::path& path) {
	std::ifstream file(path, std::ios::binary);
	if (!file)
		throw std::runtime_error("Unable to open model file for reading: " + path.string());
	const std::string json{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
	if (file.bad())
		throw std::runtime_error("Failed to read model file: " + path.string());

	NetworkModel model;
	const auto read_error = glz::read<model_read_options>(model, json);
	if (read_error)
		throw std::runtime_error("Invalid model JSON: " + glz::format_error(read_error, json));
	validate_model(model);

	NeuralNet network;
	network.input_size = model.input_size;
	network.last_layer_size = model.layers.back().neurons.size();
	network.layers.resize(model.layers.size());

	for (std::size_t i = 0; i < model.layers.size(); ++i) {
		const LayerModel& source_layer = model.layers[i];
		NeuralLayer& target_layer = network.layers[i];
		target_layer.layer_size = static_cast<int>(source_layer.neurons.size());
		target_layer.input_size = static_cast<int>(source_layer.input_size);
		target_layer.neurons = std::make_unique<Neuron[]>(source_layer.neurons.size());

		for (std::size_t j = 0; j < source_layer.neurons.size(); ++j) {
			const NeuronModel& source = source_layer.neurons[j];
			Neuron& target = target_layer.neurons[j];
			target.inputs = target_layer.input_size;
			target.constant = source.bias_input;
			target.ni = source.learning_rate;
			target.inertia = source.momentum_coefficient;
			target.w = source.weights;
			target.momentum = source.momentum;
			if (target.constant > 0.0) {
				target.w.push_back(source.bias_weight);
				target.momentum.push_back(source.bias_momentum);
			}
			target.x.assign(target.w.size(), 0.0);
			if (target.setActivation(source.activation) != 0)
				throw std::runtime_error("Unable to restore activation: " + source.activation);
		}
	}

	return network;
}
