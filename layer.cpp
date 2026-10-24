/*
 * File layer.cpp
 */

#include "layer.h"

#include <stdexcept>
#include <vector>

int NeuralLayer::Initialize (int nn, int ninputs, double min_weights, double max_weights, int act, double add, double eta)	{
	
	int i;
	if (nn <= 0 || ninputs <= 0)
		return -1;
	
	layer_size = nn;
	input_size = ninputs;
	
	neurons = std::make_unique<Neuron[]>(layer_size);

	for (i=0; i<layer_size; i++)
		if (neurons[i].Initialize(ninputs,min_weights,max_weights,act,add,eta)==-1)
			return -1;
	
return 0; // ok
}

/*
 * returns this layer size
 */
int NeuralLayer::getLayerSize ()	const {

	return layer_size;

}

/*
 * trains a layer
 */
// input is vector of incoming data. 
int NeuralLayer::Train(std::span<const double> input) {
	if (input.size() != static_cast<std::size_t>(input_size))
		throw std::invalid_argument("NeuralLayer input size does not match initialized input count");

	int i;
	for (i=0;i<layer_size;i++)	
		neurons[i].Train(input);

	return 0;	
	
}

// l, is a previous layer
int NeuralLayer::Train(const NeuralLayer& previous_layer) {

	int i,j;

	i = previous_layer.getLayerSize();

	std::vector<double> input(static_cast<std::size_t>(i));

	for (j=0;j<i;j++)	 // preparing vector
		input[static_cast<std::size_t>(j)] = previous_layer.getOutput(j);

	Train(input);

	return 0;
	
}

double NeuralLayer::getOutput(int i)	const{

	return neurons[i].getOutput();

}

void NeuralLayer::Modify ()	{

	int i;

	for (i=0;i<layer_size;i++)
		neurons[i].Modify();

	return;	

}


double NeuralLayer::getNeuronDelta(int i)	const{

	return neurons[i].getDelta();

}

// last layer count delta
void NeuralLayer::countLastDelta(std::span<const double> desired, std::span<double> error) {
		if (desired.size() != static_cast<std::size_t>(layer_size) ||
		    error.size() != static_cast<std::size_t>(layer_size))
			throw std::invalid_argument("Desired output and error sizes must match the layer size");

		int i;

		for (i=0;i<layer_size;i++)	{
			error[static_cast<std::size_t>(i)] =
				desired[static_cast<std::size_t>(i)] - neurons[i].getOutput();
			neurons[i].countDelta(error[static_cast<std::size_t>(i)]);
		}

return;
}



void NeuralLayer::countDelta(const NeuralLayer& next_layer) {
	if (next_layer.input_size != layer_size)
		throw std::invalid_argument("Next layer input size does not match this layer size");

	int i,j;
	double weighted_delta;

	for (i=0;i<layer_size;i++)	{ // for each neuron at this layer
		weighted_delta=0;
		for (j=0;j<next_layer.getLayerSize();j++)
			weighted_delta += next_layer.getNeuronWeight(j, i) * next_layer.getNeuronDelta(j);

		neurons[i].countDelta(weighted_delta);
		}

	return;
}

double NeuralLayer::getNeuronWeight(int n, int w)	const {

	return neurons[n].getWeight(w);
	
}

int NeuralLayer::setActivation (int act) {

	for (int i=0;i<layer_size;i++)
		if (neurons[i].setActivation(act) == -1)
			return -1;

	return 0;
		
}

int NeuralLayer::setActivation (double (*act) (double), double (*prim) (double))	{

	for (int i=0;i<layer_size;i++)
		if (neurons[i].setActivation(act,prim) == -1)
			return -1;
		
	return 0;

}

void NeuralLayer::setEta (double eta)	{

	for (int i=0; i<layer_size; i++)
		neurons[i].setEta (eta);
	
}
