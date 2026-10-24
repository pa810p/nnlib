/*
 * File : activation.h
 */
#ifndef _ACTIVATION_H
#define _ACTIVATION_H

#include <math.h>

#define ACT_SIGMOID	1
#define ACT_LINEAR	2
#define ACT_JUMP		3
#define ACT_SIGMOID_BI	4

double act_sigmoid (double);
double act_sigmoid_prim (double);
double act_sigmoid_bi (double);
double act_sigmoid_bi_prim (double);
double act_linear (double);
double act_linear_prim (double);
double act_jump (double);
double act_jump_prim (double);

#endif
