/*
 * File activation.cpp
 */

#include "activation.h"

// unipolar sigmoid activation function
double act_sigmoid ( double data )  {

	return (double) (1.0 / (1.0 + exp(-data)));
	
}


double act_sigmoid_prim (double data) {

	return (double) (data*(1.0-data));
	
}

// bipolar sigmoid activation function
double act_sigmoid_bi (double data)	{

	return (double) (2.0 / (1.0 + exp (-data))-1.0);
	
}


double act_sigmoid_bi_prim (double data)	{

	return (double) ((1.0 - data*data)/2);

}	

// linear activation function
double act_linear (double data)  {
   
	return (double) data;
	
}


double act_linear_prim (double data)   {
   
	(void)data;
	return (double) 1;
	
}


// jump unipolar activation function
double act_jump (double data) {
   
	return (data<0) ? 0 : 1;
	
}


double act_jump_prim (double data)  {
   
	(void)data;
	return 0;
	
}

