/*
 * File layer.h
 */
#ifndef _LAYER_H
#define _LAYER_H


#include <memory>
#include <span>
#include "neuron.h"

class NeuralLayer {
	private:
		std::unique_ptr<Neuron[]> neurons;
		int layer_size = 0;
		int input_size = 0;
		
	public:
		NeuralLayer() = default;
		~NeuralLayer() = default;
		NeuralLayer(const NeuralLayer&) = delete;
		NeuralLayer& operator=(const NeuralLayer&) = delete;
		NeuralLayer(NeuralLayer&&) noexcept = default;
		NeuralLayer& operator=(NeuralLayer&&) noexcept = default;
		int Initialize(int, int, double, double, int, double, double);
		int getLayerSize () const;
		int Train(std::span<const double> input);
		int Train(const NeuralLayer& previous_layer);

		void Modify ();
		double getNeuronWeight (int, int) const; // neuron, weight
		double getNeuronDelta (int) const;
		double getOutput (int) const; // neuron
		void countLastDelta(std::span<const double> desired, std::span<double> error);
		void countDelta(const NeuralLayer& next_layer);
		int setActivation (int);
		int setActivation (double (*) (double), double (*) (double));
		void setEta (double);
	};

#endif	
