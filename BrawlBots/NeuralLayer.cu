#include "NeuralLayer.h"
#include "Neuron.h"
#include <vector>

void NLPushInput(SNeuralLayer layer, std::vector<float> input) {
	for (int i = 0; i < layer.neurons.size; i++) {
		NeuronPushInput(layer.neurons[i], input);
	}
}
__device__ void NLPushInput(SNeuralLayer layer, thrust::device_vector<float> input) {
	for (int i = 0; i < layer.neurons.size; i++) {
		NeuronPushInput(layer.neurons[i], input);
	}
}

void NLFetchInput(SNeuralLayer layer, FetchType t) {
	for (int i = 0; i < layer.neurons.size; i++) {
		NeuronFetchInputs(layer.neurons[i], t);
	}
}
__device__ void NLFetchInput(SNeuralLayer layer, FetchType t) {
	for (int i = 0; i < layer.neurons.size; i++) {
		NeuronFetchInputs(layer.neurons[i], t);
	}
}

std::vector<float> NLGetValue(SNeuralLayer layer, FetchType t) {
	std::vector<float> values;
	for(int i = 0; i < layer.neurons.size; i++) {
		values.push_back(NeuronGetValue(layer.neurons[i], t));
	}
	return values;
}

__device__ thrust::device_vector<float> NLGetOutput(SNeuralLayer nl, thrust::device_vector<float> input, FetchType t) {
	std::vector<float> values;
	for (int i = 0; i < nl.neurons.size; i++) {
		values.push_back(NeuronGetValue(nl.neurons[i], input, t));
	}
	return values;	
}

SNeuralLayer NLCopy(SNeuralLayer nl) {
	SNeuralLayer nnl = SNeuralLayer(nl.neurons.size);
	for (int i = 0; i < nl.neurons.size; i++) {
		nnl.neurons[i].w = nl.neurons[i].w;
		nnl.neurons[i].thresh = nl.neurons[i].thresh;
	}
	return nnl;
}
SNeuralLayer NLCopy(SNeuralLayer nl, SNeuralLayer connectedLayer) {
	SNeuralLayer nnl = SNeuralLayer(nl.neurons.size, connectedLayer);
	for (int i = 0; i < nl.neurons.size; i++) {
		nnl.neurons[i].w = nl.neurons[i].w;
		nnl.neurons[i].thresh = nl.neurons[i].thresh;
	}
	return nnl;
}

void NLMutate(SNeuralLayer nl) {
	for (int i = 0; i < nl.neurons.size; i++) {
		NeuronMutate(nl.neurons[i]);
	}
}