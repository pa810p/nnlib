/*
 * file neuralnet.h
 */

#ifndef _NEURALNET_H
#define _NEURALNET_H


#include <cstddef>
#include <span>
#include <vector>
#include "layer.h"

class NeuralNet {
	private:
		std::size_t input_size = 0;
		std::size_t last_layer_size = 0;
		std::vector<NeuralLayer> layers;
		double TrainEpoch(int, std::span<const double>, std::span<const double>);
	public:
		NeuralNet(std::span<const int> layer_sizes, int inputs,
		          double min_weights, double max_weights, double eta);
		NeuralNet(std::span<const int> layer_sizes, int inputs);
		~NeuralNet () = default;
		NeuralNet(const NeuralNet&) = delete;
		NeuralNet& operator=(const NeuralNet&) = delete;
		void Train(int sample_count, std::span<const double> inputs,
		           std::span<const double> outputs, double error, long epochs);
		void Check(std::span<const double> inputs, std::span<double> outputs);
		void setAllActivation (int);
		void setAllActivation (double (*) (double), double (*) (double));
		int setActivation (int, int);
		int setActivation (int, double (*) (double), double (*) (double));
		void setEta (double);
	};


#endif	
