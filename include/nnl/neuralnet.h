/*
 * file neuralnet.h
 */

#ifndef _NEURALNET_H
#define _NEURALNET_H


#include <cstddef>
#include <filesystem>
#include <functional>
#include <span>
#include <string_view>
#include <vector>
#include "layer.h"

class NeuralNet {
	public:
		using TrainingProgressCallback = std::function<void(std::size_t, double)>;
		struct TrainingOptions {
			double target_rmse = 0.0;
			long max_epochs = 0;
			TrainingProgressCallback on_epoch;
		};

	private:
		std::size_t input_size = 0;
		std::size_t last_layer_size = 0;
		std::vector<NeuralLayer> layers;
		double TrainEpoch(int, std::span<const double>, std::span<const double>);
		NeuralNet() = default;
	public:
		NeuralNet(std::span<const int> layer_sizes, int inputs,
		          double min_weights, double max_weights, double eta);
		NeuralNet(std::span<const int> layer_sizes, int inputs);
		~NeuralNet () = default;
		NeuralNet(const NeuralNet&) = delete;
		NeuralNet& operator=(const NeuralNet&) = delete;
		NeuralNet(NeuralNet&&) noexcept = default;
		NeuralNet& operator=(NeuralNet&&) noexcept = default;
		void Train(int sample_count, std::span<const double> inputs,
		           std::span<const double> outputs, double error, long epochs);
		void Train(int sample_count, std::span<const double> inputs,
		           std::span<const double> outputs, const TrainingOptions& options);
		void Check(std::span<const double> inputs, std::span<double> outputs);
		void setAllActivation (int);
		void setAllActivation (double (*) (double), double (*) (double));
		int setActivation (int, int);
		int setActivation (int, double (*) (double), double (*) (double));
		int setActivation(int, std::string_view);
		void setEta (double);
		void setMomentum(double);
		void save(const std::filesystem::path&) const;
		static NeuralNet load(const std::filesystem::path&);
	};


#endif	
