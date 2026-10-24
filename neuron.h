/*
 * File neuron.h
 */

#ifndef _NEURON_H
#define _NEURON_H

#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "activation.h"

class Neuron {
	private:
		double * w;
		double * x;
		double delta;
		double * momentum; // needed by momentum
		double z; // weighted sum
		double y; // output of the neuron

		double (*activation) (double) ; // printer to activation function
		double (*activation_prim) (double);
		
		int inputs; // number of inputs to neuron.
		double constant; // constant signal (eg. 1)
		double ni; // speed of training
		double inertia;
	public:
		Neuron ();
		~Neuron ();
		Neuron(const Neuron&) = delete;
		Neuron& operator=(const Neuron&) = delete;
		
		int Initialize (int, double, double, int, double, double); // min range, max range
		int Initialize (int);
		
		double Train (double *);
		double Modify ();
		double getWeight (int) const;
		double getDelta () const;
		void setDelta (double);
		double getOutput () const;
		double countDelta (double);
		int setActivation (double (*) (double), double (*) (double));
		int setActivation (int);
		void setMomentum (double);
		void setEta (double);
	};	

#endif
