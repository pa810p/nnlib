/*
 * File : main.cpp
 */

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <limits>
#include <stdexcept>
#include <vector>

#include "neuralnet.h"

namespace {

std::size_t checked_buffer_size(int rows, int columns) {
	const auto row_count = static_cast<std::size_t>(rows);
	const auto column_count = static_cast<std::size_t>(columns);
	if (column_count != 0 && row_count > std::numeric_limits<std::size_t>::max() / column_count)
		throw std::length_error("Requested input data is too large");
	return row_count * column_count;
}

}

int main() {
	try {
		int layer_count;
		if (std::scanf("%d", &layer_count) != 1 || layer_count <= 0) {
			std::fprintf(stderr, "Invalid layer count\n");
			return 1;
		}

		std::vector<int> layer_sizes(static_cast<std::size_t>(layer_count));
		for (int& layer_size : layer_sizes) {
			if (std::scanf("%d", &layer_size) != 1 || layer_size <= 0) {
				std::fprintf(stderr, "Invalid layer size\n");
				return 1;
			}
		}

		int each_neuron_inputs;
		if (std::scanf("%d", &each_neuron_inputs) != 1 || each_neuron_inputs <= 0) {
			std::fprintf(stderr, "Invalid input size\n");
			return 1;
		}

		std::srand(static_cast<unsigned int>(std::time(nullptr)));
		NeuralNet nn(layer_sizes, each_neuron_inputs, -0.25, 0.25, 0.6);

		int trains;
		if (std::scanf("%d", &trains) != 1 || trains <= 0) {
			std::fprintf(stderr, "Invalid number of training samples\n");
			return 1;
		}

		const std::size_t input_count = checked_buffer_size(trains, each_neuron_inputs);
		const std::size_t output_count = checked_buffer_size(trains, layer_sizes.back());
		std::vector<double> inputs(input_count);
		std::vector<double> expected_outputs(output_count);

		for (int sample = 0; sample < trains; ++sample) {
			const std::size_t input_offset =
				static_cast<std::size_t>(sample) * each_neuron_inputs;
			const std::size_t output_offset =
				static_cast<std::size_t>(sample) * layer_sizes.back();

			for (int i = 0; i < each_neuron_inputs; ++i) {
				if (std::scanf("%lf", &inputs[input_offset + static_cast<std::size_t>(i)]) != 1) {
					std::fprintf(stderr, "Invalid training input value\n");
					return 1;
				}
			}
			for (int i = 0; i < layer_sizes.back(); ++i) {
				if (std::scanf("%lf", &expected_outputs[output_offset + static_cast<std::size_t>(i)]) != 1) {
					std::fprintf(stderr, "Invalid training output value\n");
					return 1;
				}
			}
		}

		for (int sample = 0; sample < trains; ++sample) {
			const std::size_t input_offset =
				static_cast<std::size_t>(sample) * each_neuron_inputs;
			const std::size_t output_offset =
				static_cast<std::size_t>(sample) * layer_sizes.back();

			for (int i = 0; i < each_neuron_inputs; ++i)
				std::printf(" %f ", inputs[input_offset + static_cast<std::size_t>(i)]);
			std::printf("=>");
			for (int i = 0; i < layer_sizes.back(); ++i)
				std::printf(" %f", expected_outputs[output_offset + static_cast<std::size_t>(i)]);
			std::printf("\n");
		}

		nn.Train(trains, inputs, expected_outputs, 0.000001, 50000);

		int checks;
		if (std::scanf("%d", &checks) != 1 || checks < 0) {
			std::fprintf(stderr, "Invalid number of checks\n");
			return 1;
		}
		std::printf("checks=%d\n", checks);

		std::vector<double> check_input(static_cast<std::size_t>(each_neuron_inputs));
		std::vector<double> output(static_cast<std::size_t>(layer_sizes.back()));
		for (int sample = 0; sample < checks; ++sample) {
			for (double& value : check_input) {
				if (std::scanf("%lf", &value) != 1) {
					std::fprintf(stderr, "Invalid check input value\n");
					return 1;
				}
			}

			nn.Check(check_input, output);

			std::printf("Checked :");
			for (const double value : output)
				std::printf(" %f ", value);
			std::printf("\n");
		}
	} catch (const std::exception& error) {
		std::fprintf(stderr, "%s\n", error.what());
		return 1;
	}

	return 0;
}
