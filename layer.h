/*
 * File layer.h
 */
#ifndef _LAYER_H
#define _LAYER_H


#include <stdio.h>
#include "neuron.h"

class NeuralLayer {
	private:
		Neuron * neurons;
		int layer_size;
		
	public:
		NeuralLayer();
		~NeuralLayer();
		NeuralLayer(const NeuralLayer&) = delete;
		NeuralLayer& operator=(const NeuralLayer&) = delete;
		int Initialize(int, int, double, double, int, double, double);
		int getLayerSize () const;
		int Train (double *); // Train vector on input
		int Train (NeuralLayer *); // Train, prev layer on input

		void Modify ();
		double getNeuronWeight (int, int) const; // neuron, weight
		double getNeuronDelta (int) const;
		double getOutput (int) const; // neuron
		void countLastDelta (double *, double *); // desired[], error[]
		void countDelta(NeuralLayer *);
		int setActivation (int);
		int setActivation (double (*) (double), double (*) (double));
		void setEta (double);
	};

#endif	
