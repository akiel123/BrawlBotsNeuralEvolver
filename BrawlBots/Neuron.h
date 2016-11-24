#pragma once
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <thrust\device_vector.h>

#define WEIGHT_MUTATION_FACTOR 0.05

enum class FetchType {Binary, Sigmoid, Linear};


struct HNeuron {
	int nConnections;
	int thresh;
	float value;
	std::vector<HNeuron> *c;
	std::vector<float> *w;
	std::vector<float> *input;

	HNeuron(int numberOfConnections) {
		nConnections = numberOfConnections;
		c->reserve(nConnections);
		w->reserve(nConnections);
		input->reserve(nConnections);
	}
	HNeuron(std::vector<HNeuron> *connections) {
		c = connections;
		nConnections = connections->size();
		w->reserve(nConnections);
		input->reserve(nConnections);
		for (int i = 0; i < c->size(); i++) {
			(*w)[i] = rand() / RAND_MAX;
		}
	}
	HNeuron(std::vector<HNeuron> *connections, std::vector<float> *weights) {
		if (connections->size() < weights->size()) {
			nConnections = connections->size();
			c = connections;
			for (int i = 0; i < nConnections; i++) {
				w->push_back((*weights)[i]);
			}
		}
		else {
			nConnections = weights->size();
			w = weights;
			for (int i = 0; i < nConnections; i++) {
				connections->push_back((*connections)[i]);
			}
		}
		input->reserve(nConnections);
	}
};

struct DNeuron {
	int nConnections;
	int thresh;
	float value;
	DNeuron *c;
	float *w;
	float *input;

	DNeuron() {}
	DNeuron(int numberOfConnections) {
		nConnections = numberOfConnections;
	}
	DNeuron(DNeuron *connections, int connectionCount) {
		c = connections;
		nConnections = connectionCount;
		for (int i = 0; i < connectionCount; i++) {
			w[i] = rand() / RAND_MAX;
			input[i] = 0;
		}
	}
	DNeuron(DNeuron* connections, float *weights, int connectionCount) {
		nConnections = connectionCount;
		w = weights;
		connections = connections;
		for (int i = 0; i < connectionCount; i++) {
			input[i] = 0;
		}
	}
};


void NeuronFetchInputs(HNeuron n, FetchType t);
void HNeuronPushInput(HNeuron n, std::vector<float> input);
__device__ void DNeuronPushInput(DNeuron n, float* input, int inputCount);
__device__ float NeuronGetValue(DNeuron n, float* input, FetchType t); 
float NeuronGetValue(HNeuron n, FetchType t);
void NeuronUpdateValue(HNeuron n);
void NeuronMutate(HNeuron n);
