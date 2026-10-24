/*
 * File layer.cpp
 */

#include "layer.h"

NeuralLayer::NeuralLayer()
{
	neurons=NULL;
}

NeuralLayer::~NeuralLayer()
{
	if (neurons != NULL)
		delete [] neurons;
}


int NeuralLayer::Initialize (int nn, int ninputs, double min_weights, double max_weights, int act, double add, double eta)	{
	
	int i;
	if (nn <= 0 || ninputs <= 0)
		return -1;
	
	layer_size = nn;
	
	if ((neurons = new Neuron [layer_size])==NULL)	return -1;

	for (i=0; i<layer_size; i++)
		if ((neurons+i)->Initialize(ninputs,min_weights,max_weights,act,add,eta)==-1)
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
int NeuralLayer::Train (double * input)	{

	int i;
	for (i=0;i<layer_size;i++)	
		(neurons+i)->Train(input); // train each neuron, by input vector

	return 0;	
	
}

// l, is a previous layer
int NeuralLayer::Train (NeuralLayer * l)	{

	int i,j;

	i = l->getLayerSize();

	double * vector = new double [i];

	for (j=0;j<i;j++)	 // preparing vector
		*(vector+j)=l->getOutput(j);

	Train(vector);
	delete [] vector;

	return 0;
	
}

double NeuralLayer::getOutput(int i)	const{

	return (neurons+i)->getOutput();

}

void NeuralLayer::Modify ()	{

	int i;

	for (i=0;i<layer_size;i++)
		(neurons+i)->Modify();

	return;	

}


double NeuralLayer::getNeuronDelta(int i)	const{

	return (neurons+i)->getDelta();

}

// last layer count delta
void NeuralLayer::countLastDelta (double * desired, double * error)	{

	int i;

	for (i=0;i<layer_size;i++)	{
		*(error+i)=(*(desired+i)-(neurons+i)->getOutput());
		(neurons+i)->countDelta(*(error+i)); 
		}

return;
}



void NeuralLayer::countDelta (NeuralLayer * l)	{

	int i,j;
	double weighted_delta;

	for (i=0;i<layer_size;i++)	{ // for each neuron at this layer
		weighted_delta=0;
		for (j=0;j<l->getLayerSize();j++)	
			weighted_delta+=(l->getNeuronWeight(j,i)*l->getNeuronDelta(j)); 

		(neurons+i)->countDelta(weighted_delta);
		}

	return;
}

double NeuralLayer::getNeuronWeight(int n, int w)	const {

	return (neurons+n)->getWeight(w);
	
}

int NeuralLayer::setActivation (int act) {

	for (int i=0;i<layer_size;i++)
		if ((neurons+i)->setActivation(act) == -1)
			return -1;

	return 0;
		
}

int NeuralLayer::setActivation (double (*act) (double), double (*prim) (double))	{

	for (int i=0;i<layer_size;i++)
		if ((neurons+i)->setActivation(act,prim) == -1)
			return -1;
		
	return 0;

}

void NeuralLayer::setEta (double eta)	{

	for (int i=0; i<layer_size; i++)
		(neurons+i)->setEta (eta);
	
}
