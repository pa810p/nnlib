/*
 * File neuralnet.cpp
 */
 

#include "neuralnet.h"
#include <stdexcept>


NeuralNet::NeuralNet (int lc, int * ls, int inputs, double min_weights, double max_weights, double eta) : layers(NULL) {
// lc number of the layers
//	* ls each layer size
//	inputs first layer each neuron inputs
//	min_weights minimal randomized weights
//	max_weights maximal randomized weights
// ep maximal number of epochs

	int i;
	if (lc <= 0 || ls == NULL || inputs <= 0)
		throw std::invalid_argument("NeuralNet requires positive layer and input counts");
	for (i=0; i<lc; i++)
		if (*(ls+i) <= 0)
			throw std::invalid_argument("NeuralNet layer sizes must be positive");
	layer_count = lc;
	input_size = inputs;
	
	if ((layers = new NeuralLayer [layer_count])==NULL)	{
		printf("Error while allocate memory for layer\n");
		exit(1);
		}

	for (i=0;i<lc;i++)	{
		if (i==0)	{ // first layer
			if ((layers+i)->Initialize(*(ls+i), inputs, min_weights, max_weights, ACT_SIGMOID, 1, eta)==-1)	{
				printf("Error while initializing input layer");
				exit(1);
				};
			}
		else if (i!=lc-1){ 
			if ((layers+i)->Initialize(*(ls+i), *(ls+i-1), min_weights, max_weights, ACT_SIGMOID,1, eta)==-1) {
				printf("Error while initializing hidden layers");
				exit(1);
				}
			}
		else { // last layer
			if ((layers+i)->Initialize(*(ls+i), *(ls+i-1), min_weights, max_weights, ACT_LINEAR, 1, eta)==-1)	{
				printf("Error while initializing output layer");
				exit(1);
				}
			}	
		}
	
	last_layer = layers+layer_count-1;
	last_layer_size = last_layer->getLayerSize();

}

NeuralNet::NeuralNet (int lc, int * ls, int inputs)
	: NeuralNet(lc, ls, inputs, 0., 1., 0.5) {}

NeuralNet::~NeuralNet()
{
	if (layers!=NULL)
		delete [] layers;
}

double * NeuralNet::Check (double * inputs, double * outputs)	{

	int i;
	
	layers->Train(inputs); // trains first layer
	for (i=1;i<layer_count;i++)	{
		(layers+i)->Train(layers+i-1);
		}

	for (i=0;i<(layers+layer_count-1)->getLayerSize();i++)
		*(outputs+i)=(layers+layer_count-1)->getOutput(i);

return outputs;
}


double NeuralNet::TrainEpoch(int counts, double * inputs, double * outputs)	{

	int i,j;
	double sum=0;

	double * error = new double [last_layer_size];
	
	for (j=0;j<counts;j++)	{

		layers->Train(inputs+input_size*j); // train first layer
		
		for (i=1;i<layer_count;i++)
			(layers+i)->Train(layers+i-1);

		last_layer->countLastDelta(outputs+last_layer_size*j,error);

		for (i=layer_count-2;i>=0;i--)
			(layers+i)->countDelta(layers+i+1);

		for (i=0;i<layer_count;i++)
			(layers+i)->Modify();

		for (i=0;i<last_layer_size;i++)
			sum+=(*(error+i)**(error+i));

		}

	delete [] error;

	return sqrt(sum/(counts*last_layer_size));

}


void NeuralNet::Train (int counts, double * inputs, double * outputs, double err, long epochs)	{

	double krit;
	long i=0;

	krit = TrainEpoch(counts, inputs, outputs);
	
	while ((krit>err)&&(i<epochs))	{
		krit = TrainEpoch(counts, inputs, outputs);
		i++;
		}

	return;

}

int NeuralNet::setActivation (int i, int act)	{

	if ((i<0)||(i>=layer_count)) return -1;
	else return (layers+i)->setActivation(act);

}

int NeuralNet::setActivation (int i, double (*act) (double), double (*prim)(double))	{

	if ((i<0)||(i>=layer_count)) return -1;
	else return (layers+i)->setActivation(act,prim);

}

void NeuralNet::setAllActivation (int act)	{

	for (int i=0;i<layer_count;i++)
		(layers+i)->setActivation(act);

}

void NeuralNet::setAllActivation (double (*act) (double), double (*prim) (double))	{

	for (int i=0;i<layer_count;i++)
		(layers+i)->setActivation(act,prim);

}		

void NeuralNet::setEta (double eta)	{
	
	for (int i=0;i<layer_count;i++)	
		(layers+i)->setEta(eta);
}
