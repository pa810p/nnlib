/*
 * File activation.cpp
 */

#include <nnl/activation.h>

#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace {

std::mutex& activation_registry_mutex() {
	static std::mutex mutex;
	return mutex;
}

std::unordered_map<std::string, ActivationFunctions>& activation_registry() {
	static std::unordered_map<std::string, ActivationFunctions> registry;
	return registry;
}

}

void registerActivation(std::string id, ActivationFunction function,
						ActivationFunction derivative) {
	if (id.rfind("custom.", 0) != 0 || id.size() == 7)
		throw std::invalid_argument("Custom activation IDs must start with 'custom.' and have a name");
	if (function == nullptr || derivative == nullptr)
		throw std::invalid_argument("Activation function and derivative must not be null");

	std::lock_guard<std::mutex> lock(activation_registry_mutex());
	const auto [iterator, inserted] = activation_registry().emplace(
		std::move(id), ActivationFunctions{function, derivative});
	if (!inserted)
		throw std::invalid_argument("Activation ID is already registered: " + iterator->first);
}

std::optional<ActivationFunctions> resolveActivation(std::string_view id) {
	if (id == ACTIVATION_SIGMOID_ID)
		return ActivationFunctions{act_sigmoid, act_sigmoid_prim};
	if (id == ACTIVATION_LINEAR_ID)
		return ActivationFunctions{act_linear, act_linear_prim};
	if (id == ACTIVATION_JUMP_ID)
		return ActivationFunctions{act_jump, act_jump_prim};
	if (id == ACTIVATION_SIGMOID_BI_ID)
		return ActivationFunctions{act_sigmoid_bi, act_sigmoid_bi_prim};

	std::lock_guard<std::mutex> lock(activation_registry_mutex());
	const auto iterator = activation_registry().find(std::string(id));
	if (iterator == activation_registry().end())
		return std::nullopt;
	return iterator->second;
}

std::string_view activationId(int activation) {
	switch (activation) {
		case ACT_SIGMOID: return ACTIVATION_SIGMOID_ID;
		case ACT_LINEAR: return ACTIVATION_LINEAR_ID;
		case ACT_JUMP: return ACTIVATION_JUMP_ID;
		case ACT_SIGMOID_BI: return ACTIVATION_SIGMOID_BI_ID;
		default: return {};
	}
}

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

