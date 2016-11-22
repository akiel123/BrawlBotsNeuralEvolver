#include "Neuron.h"
#include <math.h>

void NeuronFetchInputs(SNeuron n, FetchType t) {
	for (int i = 0; i < n.c.size && i < n.input.size; i++) {
		n.input[i] = NeuronGetValue(n.c[i], t);
	}
}

void NeuronPushInput(SNeuron n, std::vector<float> input) {
	for (int i = 0; i < n.input.size && i < input.size; i++) {
		n.input[i] = input[i];
	}
}

float NeuronGetValue(SNeuron n, FetchType t) {
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

__device__ float NeuronGetValue(SNeuron n, thrust::device_vector<float> input, FetchType t) {
	float sum = 0;
	float value;
	if (n.w.size == 0) value = n.input[0];
	for (int i = 0; i < n.input.size && i < n.w.size; i++) {
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

void NeuronUpdateValue(SNeuron n) {
	float sum = 0;
	if (n.w.size == 0) n.value = n.input[0];
	for (int i = 0; i < n.input.size && i < n.w.size; i++) {
		sum += n.input[i] * n.w[i];
	}
	n.value = sum;
}

void NeuronMutate(SNeuron n) {
	for (int i = 0; i < n.w.size; i++) {
		n.w[i] += (rand() / RAND_MAX * 2 - 1) * WEIGHT_MUTATION_FACTOR;
	}
	n.thresh += (rand() / RAND_MAX * 2 - 1) * WEIGHT_MUTATION_FACTOR * n.w.size;
}

void NeuronInitialize(SNeuron n) {

}