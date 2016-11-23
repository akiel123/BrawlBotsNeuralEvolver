#pragma once
#include "Neuron.h"
#include <thrust\device_vector.h>

struct SNeuralLayer {
	std::vector<SNeuron> *neurons;

	SNeuralLayer(int nNeurons) {
		for (int i = 0; i < nNeurons; i++) {
			neurons->push_back(SNeuron(1));
		}
	}
	SNeuralLayer(int nNeurons, std::vector<SNeuron> inputs) {
		for (int i = 0; i < nNeurons; i++) {
			neurons->push_back(SNeuron(inputs));
		}
	}
	SNeuralLayer(int nNeurons, SNeuralLayer inputLayer) {
		for (int i = 0; i < nNeurons; i++) {
			neurons->push_back(SNeuron(inputLayer.neurons));
		}
	}

};

void NLPushInput(SNeuralLayer layer, std::vector<float> input);
__device__ void NLPushInput(SNeuralLayer layer, thrust::device_vector<float> input);
void NLFetchInput(SNeuralLayer layer, FetchType t);
__device__ void NLFetchInput(SNeuralLayer layer, FetchType t);
std::vector<float> NLGetValue(SNeuralLayer layer, FetchType t);
__device__ thrust::device_vector<float> NLGetOutput(SNeuralLayer nl, thrust::device_vector<float> input, FetchType t);
SNeuralLayer NLCopy(SNeuralLayer nl);
SNeuralLayer NLCopy(SNeuralLayer nl, SNeuralLayer connectedLayer);
void NLMutate(SNeuralLayer nl);