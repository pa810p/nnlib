/*
 * file neuralnet.h
 */

#ifndef _NEURALNET_H
#define _NEURALNET_H


#include <stdio.h>
#include "layer.h"

class NeuralNet {
	private:
		int layer_count;
		int input_size;
		int last_layer_size;
		NeuralLayer * layers; // pointer to layers of neuralnet
		NeuralLayer * last_layer;
		double TrainEpoch(int, double *, double *);
	public:
		NeuralNet (int, int *, int, double, double, double);
		NeuralNet (int, int *, int);
		~NeuralNet ();
		NeuralNet(const NeuralNet&) = delete;
		NeuralNet& operator=(const NeuralNet&) = delete;
		void Train (int, double *, double *, double, long);
		double * Check (double *, double *);
		void setAllActivation (int);
		void setAllActivation (double (*) (double), double (*) (double));
		int setActivation (int, int);
		int setActivation (int, double (*) (double), double (*) (double));
		void setEta (double);
	};


#endif	
