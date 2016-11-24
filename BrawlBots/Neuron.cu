#include "Neuron.h"
#include <math.h>

void NeuronFetchInputs(HNeuron n, FetchType t) {
	for (int i = 0; i < n.c->size && i < n.input->size; i++) {
		(*n.input)[i] = NeuronGetValue((*n.c)[i], t);
	}
}

void HNeuronPushInput(HNeuron n, std::vector<float> input) {
	for (int i = 0; i < n.input->size && i < input.size; i++) {
		(*n.input)[i] = input[i];
	}
}
__device__ void DNeuronPushInput(DNeuron n, float* input, int inputCount) {
	for (int i = 0; i < n.nConnections && i < inputCount; i++) {
		n.input[i] = input[i];
	}
}

float NeuronGetValue(HNeuron n, FetchType t) {
	switch (t) {
	case FetchType::Binary:
		return n.value > n.thresh;
	case FetchType::Sigmoid:
		return 1 / (1 + exp(n.value));
	case FetchType::Linear:
		return n.value;
	default:
		return 0;
	}
}


__device__ float NeuronGetValue(DNeuron n, float* input, FetchType t) {
	float sum = 0;
	float value;
	if (n.nConnections == 0) value = n.input[0];
	for (int i = 0; i < n.nConnections; i++) {
		sum += n.input[i] * n.w[i];
	}
	value = sum;
	switch (t) {
	case FetchType::Binary:
		return value > n.thresh;
	case FetchType::Sigmoid:
		return 1 / (1 + exp(value));
	case FetchType::Linear:
		return value;
	default:
		return 0;
	}
}

void NeuronUpdateValue(HNeuron n) {
	float sum = 0;
	if (n.w->size == 0) n.value = (*n.input)[0];
	for (int i = 0; i < n.input->size && i < n.w->size; i++) {
		sum += (*n.input)[i] * (*n.w)[i];
	}
	n.value = sum;
}

void NeuronMutate(HNeuron n) {
	for (int i = 0; i < n.w->size; i++) {
		(*n.w)[i] += (rand() / RAND_MAX * 2 - 1) * WEIGHT_MUTATION_FACTOR;
	}
	n.thresh += (rand() / RAND_MAX * 2 - 1) * WEIGHT_MUTATION_FACTOR * (n.w->size);
}

void NeuronInitialize(HNeuron n) {

}