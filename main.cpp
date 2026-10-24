/*
 * File : main.cpp
 */
#include <stdio.h>

#include "neuralnet.h"

int main ()	{

	int layer_count;
	int each_neuron_inputs;
	int last_layer_size;
	int trains;
	int checks;
	int j=0;
	int i;

	if (scanf("%d", &layer_count) != 1 || layer_count <= 0) {
		fprintf(stderr, "Invalid layer count\n");
		return 1;
	}
	int * ls = new int [layer_count];
	
	for (i=0;i<layer_count;i++) {
		if (scanf("%d", ls+i) != 1 || *(ls+i) <= 0) {
			fprintf(stderr, "Invalid layer size\n");
			delete [] ls;
			return 1;
		}
	}

	last_layer_size=*(ls+layer_count-1);

	if (scanf("%d", &each_neuron_inputs) != 1 || each_neuron_inputs <= 0) {
		fprintf(stderr, "Invalid input size\n");
		delete [] ls;
		return 1;
	}

	long seed=time((time_t)NULL);
	srand(seed);

	NeuralNet * nn = new NeuralNet(layer_count, ls, each_neuron_inputs, (double)-0.25, (double)0.25, 0.6);

	if (scanf("%d", &trains) != 1 || trains <= 0) {
		fprintf(stderr, "Invalid number of training samples\n");
		delete nn;
		delete [] ls;
		return 1;
	}
	
	double * inputs = new double [each_neuron_inputs*trains];
	double * output = new double [last_layer_size*trains];
	
	for (i=0;i<trains;i++) {
		for (j=0; j<each_neuron_inputs; j++)
			if (scanf("%lf",inputs+(i*each_neuron_inputs)+j) != 1) {
				fprintf(stderr, "Invalid training input value\n");
				delete [] output;
				delete [] inputs;
				delete nn;
				delete [] ls;
				return 1;
			}
		for (j=0; j<last_layer_size; j++)
			if (scanf("%lf",output+(i*last_layer_size)+j) != 1) {
				fprintf(stderr, "Invalid training output value\n");
				delete [] output;
				delete [] inputs;
				delete nn;
				delete [] ls;
				return 1;
			}
		}	

	for (i=0;i<trains;i++) {
		for (j=0; j<each_neuron_inputs; j++)
			printf(" %f ",*(inputs+(i*each_neuron_inputs)+j));
		printf("=>");	
		for (j=0; j<last_layer_size; j++)
			printf(" %f",*(output+(i*last_layer_size)+j));
		printf("\n");
		}

		nn->Train(trains,inputs,output,0.000001,50000);

	if (scanf("%d", &checks) != 1 || checks < 0) {
		fprintf(stderr, "Invalid number of checks\n");
		delete [] output;
		delete [] inputs;
		delete nn;
		delete [] ls;
		return 1;
	}
	printf ("checks=%d\n",checks);
	
	double * back = new double [last_layer_size*checks];
	
	for (i=0;i<checks;i++)	{
		for (j=0; j<each_neuron_inputs; j++)
			if (scanf("%lf",inputs+j) != 1) {
				fprintf(stderr, "Invalid check input value\n");
				delete [] back;
				delete [] output;
				delete [] inputs;
				delete nn;
				delete [] ls;
				return 1;
			}
			
		nn ->Check (inputs,back);
		
		printf ("Checked :");
		for (j=0;j<last_layer_size;j++)
			printf(" %f ", *(back+j));
		printf("\n");	
		}

	delete [] back;
	delete [] output;
	delete [] inputs;
	delete nn;
	delete [] ls;

return 0;
}

