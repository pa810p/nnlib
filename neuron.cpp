/*
 * File neuron.cpp
 */

#include "neuron.h"

#include <cstdlib>
#include <stdexcept>

/*
 * initialize neuron : random weights, at range min to max, activation function and constant input.
 */
int Neuron::Initialize (int numb, double minimum, double maximum, int act, double add, double speed)      {

	double min;
	double max;
	int a;

	ni = speed;

	a = (add>0) ? 1 : 0;
	
	constant = add;
	inertia = 0;

	y=z=0;
	
	inputs=numb;
	w.resize(inputs+a);
	x.resize(inputs+a);
	momentum.assign(inputs+a, 0.0);

	if (minimum>maximum) { // afraid of dump users... happen...
		min=maximum;
		max=minimum;
		}
	else {
		max=maximum;
		min=minimum;
		}

		// randomizing weights loop
		for (int i=0; i<inputs+a; i++) {
			double r = static_cast<double>(std::rand()) / (static_cast<double>(RAND_MAX) + 1.0);
			w[i] = min + ((max - min) * r);
		}

	if (setActivation(act) == -1)
		return -1;

return 0;
}

/*
 * initialize random weights, from 0-1
 */
int Neuron::Initialize (int numb)	{

	return this->Initialize(numb,0,1,ACT_SIGMOID,0,0.5);

}

double Neuron::Train(std::span<const double> in) {
	if (in.size() != static_cast<std::size_t>(inputs))
		throw std::invalid_argument("Neuron input size does not match initialized input count");

	z=0;

	for (int i=0;i<inputs;i++)	{
		x[i]=in[static_cast<std::size_t>(i)];
		z+=(x[i]*w[i]);
		}
	
	if (constant)	z+=(w[inputs]*constant);
	
	y = activation (z);
	return y;
}


double Neuron::Modify () {

	int i;
		
	if (constant) {
		w[inputs]+=(ni*delta*constant)+momentum[inputs]*inertia;
		momentum[inputs] = (ni*delta*constant);
	}

	for (i=0;i<inputs;i++) {
		w[i] += (ni*delta*x[i])+momentum[i]*inertia;//*y*(1-y);
		momentum[i] = (ni*delta*x[i]);//+*(momentum+i)*inertia;
		}
	
return y;
}


double Neuron::getWeight (int i)	const {

	return w[i];

}


double Neuron::getOutput()	const {

	return y;

}


double Neuron::countDelta (double weighted_delta)	{

//	printf("Next layer delta = %2.10f\n",weighted_delta);
	return delta = activation_prim(y)*weighted_delta;

}

double Neuron::getDelta ()	const{

	return delta;

}

void Neuron::setDelta ( double d )	{
	
	delta = d;

}	

int Neuron::setActivation ( double (*act) (double), double (*prim) (double))	{

	if (act == NULL || prim == NULL)
		return -1;
	activation = act;
	activation_prim = prim;
	return 0;

}

int Neuron::setActivation (int act) {

	switch (act)	{
		case ACT_SIGMOID:
			activation = &act_sigmoid;
			activation_prim = act_sigmoid_prim;
			break;
		case ACT_LINEAR:
			activation = act_linear;
			activation_prim = act_linear_prim;
			break;
		case ACT_JUMP:
			activation = act_jump;
			activation_prim = act_jump_prim;
			break;
		case ACT_SIGMOID_BI:
			activation = &act_sigmoid_bi;
			activation_prim = act_sigmoid_bi_prim;
			break;
		default:
			return -1;
		}
	return 0;
}	

void Neuron::setMomentum (double mom)	{

	inertia = mom;

}

void Neuron::setEta (double eta)	{

	ni = eta;

}
