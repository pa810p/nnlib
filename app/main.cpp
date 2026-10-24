/*
 * File : main.cpp
 */

#include <cstdlib>
#include <ctime>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <nnl/neuralnet.h>
#include "logger.h"
#include "training_data.h"

namespace {

std::vector<double> flatten_training_inputs(const TrainingData& data) {
	std::vector<double> inputs;
	inputs.reserve(data.training.size() * data.input_size);
	for (const TrainingSample& sample : data.training)
		inputs.insert(inputs.end(), sample.input.begin(), sample.input.end());
	return inputs;
}

std::vector<double> flatten_training_targets(const TrainingData& data) {
	const auto output_size = static_cast<std::size_t>(data.layer_sizes.back());
	std::vector<double> targets;
	targets.reserve(data.training.size() * output_size);
	for (const TrainingSample& sample : data.training)
		targets.insert(targets.end(), sample.target.begin(), sample.target.end());
	return targets;
}

} // namespace

int main() {
	try {
		const char* configured_level = std::getenv("NNLIB_LOG_LEVEL");
		Logger logger(std::clog, Logger::parse_level(
			configured_level == nullptr ? "INFO" : configured_level));
		const std::string json{std::istreambuf_iterator<char>(std::cin),
		                       std::istreambuf_iterator<char>()};
		const TrainingData data = parse_training_data(json);
		const int sample_count = static_cast<int>(data.training.size());
		const int input_size = static_cast<int>(data.input_size);
		const std::size_t output_size = static_cast<std::size_t>(data.layer_sizes.back());
		const std::vector<double> inputs = flatten_training_inputs(data);
		const std::vector<double> expected_outputs = flatten_training_targets(data);

		std::srand(static_cast<unsigned int>(std::time(nullptr)));
		NeuralNet nn(data.layer_sizes, input_size, -0.25, 0.25, 0.6);
		std::cout << "Training data: " << data.name << '\n';
		for (std::size_t sample = 0; sample < data.training.size(); ++sample) {
			std::cout << "  ";
			for (const double value : data.training[sample].input)
				std::cout << value << ' ';
			std::cout << "=> ";
			for (const double value : data.training[sample].target)
				std::cout << value << ' ';
			std::cout << '\n';
		}
		std::size_t completed_epochs = 0;
		double final_error = 0.0;
		const NeuralNet::TrainingOptions training_options{
			.target_rmse = 0.000001,
			.max_epochs = 50000,
			.on_epoch = [&](std::size_t epoch, double error) {
			         completed_epochs = epoch;
			         final_error = error;
			         if (!logger.enabled(LogLevel::DEBUG))
				         return;
			         std::ostringstream message;
			         message << "epoch=" << epoch << " training_rmse="
					         << std::fixed << std::setprecision(8) << error;
			         logger.debug(message.str());
			}
		};
		nn.Train(sample_count, inputs, expected_outputs, training_options);
		std::ostringstream summary;
		summary << "training complete: epochs=" << completed_epochs
		        << " training_rmse=" << std::fixed << std::setprecision(8) << final_error;
		logger.info(summary.str());

		constexpr int output_precision = 4;
		constexpr double output_scale = 10000.0;
		std::cout << std::fixed << std::setprecision(output_precision);
		std::vector<double> output(output_size);
		for (const std::vector<double>& check_input : data.checks) {
			nn.Check(check_input, output);
			std::cout << "Checked :";
			for (const double value : output) {
				const double rounded_value = std::round(value * output_scale) / output_scale;
				std::cout << ' ' << (rounded_value == 0.0 ? 0.0 : rounded_value);
			}
			std::cout << '\n';
		}
	} catch (const std::exception& error) {
		std::cerr << error.what() << '\n';
		return 1;
	}

	return 0;
}
