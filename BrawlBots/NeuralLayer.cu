#include "NeuralLayer.h"
#include "Neuron.h"
#include <vector>

void NLPushInput(HNeuralLayer layer, std::vector<float> input) {
	for (int i = 0; i < layer.neurons->size; i++) {
		HNeuronPushInput((*layer.neurons)[i], input);
	}
}
__device__ void NLPushInput(DNeuralLayer layer, float* input, int inputCount) {
	for (int i = 0; i < layer.nNeurons; i++) {
		DNeuronPushInput(layer.neurons[i], input, inputCount);
	}
}

void NLFetchInput(HNeuralLayer layer, FetchType t) {
	for (int i = 0; i < layer.neurons->size; i++) {
		NeuronFetchInputs((*layer.neurons)[i], t);
	}
}
/*__device__ void NLFetchInput(DNeuralLayer layer, FetchType t) {
	for (int i = 0; i < layer.nNeurons; i++) {
		NeuronFetchInputs(layer.neurons[i], t);
	}
}*/

std::vector<float> NLGetValue(HNeuralLayer layer, FetchType t) {
	std::vector<float> values;
	for(int i = 0; i < layer.neurons->size(); i++) {
		values.push_back(NeuronGetValue((*layer.neurons)[i], t));
	}
	return values;
}

__device__ float* NLGetOutput(DNeuralLayer nl, float* input, FetchType t) {
	float* values;
	for (int i = 0; i < nl.nNeurons; i++) {
		values[i] = (NeuronGetValue(nl.neurons[i], input, t));
	}
	return values;	
}

HNeuralLayer NLCopy(HNeuralLayer nl) {
	HNeuralLayer nnl = HNeuralLayer(nl.neurons->size);
	for (int i = 0; i < nl.neurons->size(); i++) {
		(*nnl.neurons)[i].w = (*nl.neurons)[i].w;
		(*nnl.neurons)[i].thresh = (*nl.neurons)[i].thresh;
	}
	return nnl;
}
HNeuralLayer NLCopy(HNeuralLayer nl, HNeuralLayer connectedLayer) {
	HNeuralLayer nnl = HNeuralLayer(nl.neurons->size(), connectedLayer);
	for (int i = 0; i < nl.neurons->size(); i++) {
		(*nnl.neurons)[i].w = (*nl.neurons)[i].w;
		(*nnl.neurons)[i].thresh = (*nl.neurons)[i].thresh;
	}
	return nnl;
}

void NLMutate(HNeuralLayer nl) {
	for (int i = 0; i < nl.neurons->size; i++) {
		NeuronMutate((*nl.neurons)[i]);
	}
}