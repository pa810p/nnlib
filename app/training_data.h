#pragma once

#include <glaze/json.hpp>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

struct TrainingSample {
	std::vector<double> input;
	std::vector<double> target;
};

struct TrainingData {
	std::string name;
	std::vector<int> layer_sizes;
	std::size_t input_size = 0;
	std::vector<TrainingSample> training;
	std::vector<std::vector<double>> checks;
};

inline TrainingData parse_training_data(std::string_view json) {
	constexpr glz::opts read_options{.error_on_missing_keys = true};
	TrainingData data;
	const auto error = glz::read<read_options>(data, json);
	if (error)
		throw std::runtime_error("Invalid training data JSON: " +
		                         glz::format_error(error, json));

	if (data.name.empty() || data.layer_sizes.empty() || data.input_size == 0 ||
	    data.input_size > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
	    data.training.empty() ||
	    data.training.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
		throw std::runtime_error("Training data has invalid network or sample dimensions");
	for (const int layer_size : data.layer_sizes)
		if (layer_size <= 0)
			throw std::runtime_error("Training data layer sizes must be positive");

	const auto output_size = static_cast<std::size_t>(data.layer_sizes.back());
	for (const TrainingSample& sample : data.training)
		if (sample.input.size() != data.input_size || sample.target.size() != output_size)
			throw std::runtime_error("Training sample dimensions do not match the network");
	for (const std::vector<double>& check : data.checks)
		if (check.size() != data.input_size)
			throw std::runtime_error("Check input dimensions do not match the network");

	return data;
}

inline TrainingData load_training_data(const std::filesystem::path& path) {
	std::ifstream file(path, std::ios::binary);
	if (!file)
		throw std::runtime_error("Unable to open training data file: " + path.string());
	const std::string json{std::istreambuf_iterator<char>(file),
	                       std::istreambuf_iterator<char>()};
	if (file.bad())
		throw std::runtime_error("Failed to read training data file: " + path.string());
	return parse_training_data(json);
}