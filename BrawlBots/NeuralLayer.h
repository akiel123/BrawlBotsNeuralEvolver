#pragma once
#include "Neuron.h"
#include <thrust\device_vector.h>

struct HNeuralLayer {
	std::vector<HNeuron> *neurons;

	HNeuralLayer(int nNeurons) {
		for (int i = 0; i < nNeurons; i++) {
			neurons->push_back(HNeuron(1));
		}
	}
	HNeuralLayer(int nNeurons, std::vector<HNeuron>* inputs) {
		for (int i = 0; i < nNeurons; i++) {
			neurons->push_back(HNeuron(inputs));
		}
	}
	HNeuralLayer(int nNeurons, HNeuralLayer inputLayer) {
		for (int i = 0; i < nNeurons; i++) {
			neurons->push_back(HNeuron(inputLayer.neurons));
		}
	}

};

struct DNeuralLayer {
	DNeuron* neurons;
	int nNeurons;

	DNeuralLayer(int neuronCount) {
		nNeurons = neuronCount;
		for (int i = 0; i < nNeurons; i++) {
			neurons[i] = DNeuron(1);
		}
	}
	DNeuralLayer(int neuronCount, DNeuron* inputs, int inputCount) {
		nNeurons = neuronCount;
		for (int i = 0; i < nNeurons; i++) {
			neurons[i] = DNeuron(inputs, inputCount);
		}
	}
	DNeuralLayer(int nNeurons, DNeuralLayer inputLayer) {
		for (int i = 0; i < nNeurons; i++) {
			neurons[i] = DNeuron(inputLayer.neurons, inputLayer.nNeurons);
		}
	}

};

void NLPushInput(HNeuralLayer layer, std::vector<float> input);
__device__ void NLPushInput(DNeuralLayer layer, float* input, int inputCount);
void NLFetchInput(HNeuralLayer layer, FetchType t);
//__device__ void NLFetchInput(DNeuralLayer layer, FetchType t);
std::vector<float> NLGetValue(HNeuralLayer layer, FetchType t);
__device__ float* NLGetOutput(DNeuralLayer nl, float* input, FetchType t);
HNeuralLayer NLCopy(HNeuralLayer nl);
HNeuralLayer NLCopy(HNeuralLayer nl, HNeuralLayer connectedLayer);
void NLMutate(HNeuralLayer nl);