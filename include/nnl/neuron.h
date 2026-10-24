/*
 * File neuron.h
 */

#ifndef _NEURON_H
#define _NEURON_H

#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "activation.h"

class Neuron {
	private:
		std::vector<double> w;
		std::vector<double> x;
		double delta = 0;
		std::vector<double> momentum; // needed by momentum
		double z = 0; // weighted sum
		double y = 0; // output of the neuron

		double (*activation) (double) = nullptr; // printer to activation function
		double (*activation_prim) (double) = nullptr;
		std::string activation_id;
		
		int inputs = 0; // number of inputs to neuron.
		double constant = 0; // constant signal (eg. 1)
		double ni = 0; // speed of training
		double inertia = 0;
		friend class NeuralNet;
		friend class NeuralLayer;
	public:
		Neuron() = default;
		~Neuron() = default;
		Neuron(const Neuron&) = delete;
		Neuron& operator=(const Neuron&) = delete;
		
		int Initialize (int, double, double, int, double, double); // min range, max range
		int Initialize (int);
		
		double Train (std::span<const double>);
		double Modify ();
		double getWeight (int) const;
		double getDelta () const;
		void setDelta (double);
		double getOutput () const;
		double countDelta (double);
		int setActivation (double (*) (double), double (*) (double));
		int setActivation (int);
		int setActivation(std::string_view);
		void setMomentum (double);
		void setEta (double);
	};	

#endif
