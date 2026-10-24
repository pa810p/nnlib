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

#include <optional>
#include <string>
#include <string_view>

using ActivationFunction = double (*)(double);

struct ActivationFunctions {
	ActivationFunction function;
	ActivationFunction derivative;
};

inline constexpr std::string_view ACTIVATION_SIGMOID_ID = "sigmoid";
inline constexpr std::string_view ACTIVATION_LINEAR_ID = "linear";
inline constexpr std::string_view ACTIVATION_JUMP_ID = "jump";
inline constexpr std::string_view ACTIVATION_SIGMOID_BI_ID = "sigmoid_bipolar";

void registerActivation(std::string id, ActivationFunction function,
						ActivationFunction derivative);
std::optional<ActivationFunctions> resolveActivation(std::string_view id);
std::string_view activationId(int activation);

double act_sigmoid (double);
double act_sigmoid_prim (double);
double act_sigmoid_bi (double);
double act_sigmoid_bi_prim (double);
double act_linear (double);
double act_linear_prim (double);
double act_jump (double);
double act_jump_prim (double);

#endif
