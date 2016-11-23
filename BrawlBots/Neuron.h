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
	std::vector<SNeuron> c;
	std::vector<float> w;
	std::vector<float> input;

	SNeuron(int numberOfConnections) {
		nConnections = numberOfConnections;
		c.reserve(nConnections);
		w.reserve(nConnections);
		input.reserve(nConnections);
	}
	SNeuron(std::vector<SNeuron> connections) {
		c = connections;
		nConnections = connections.size();
		w.reserve(nConnections);
		input.reserve(nConnections);
		for (int i = 0; i < c.size(); i++) {
			w[i] = rand() / RAND_MAX;
		}
	}
	SNeuron(std::vector<SNeuron> connections, std::vector<float> weights) {
		if (connections.size() < weights.size()) {
			nConnections = connections.size();
			c = connections;
			for (int i = 0; i < nConnections; i++) {
				w.push_back(weights[i]);
			}
		}
		else {
			nConnections = weights.size();
			w = weights;
			for (int i = 0; i < nConnections; i++) {
				connections.push_back(connections[i]);
			}
		}
		input.reserve(nConnections);
	}
};

struct DNeuron {
	int nConnections;
	int thresh;
	float value;
	thrust::device_vector<DNeuron> c;
	thrust::device_vector<float> w;
	thrust::device_vector<float> input;

	DNeuron(int numberOfConnections) {
		nConnections = numberOfConnections;
		c.reserve(nConnections);
		w.reserve(nConnections);
		input.reserve(nConnections);
	}
	DNeuron(thrust::device_vector<DNeuron> connections) {
		c = connections;
		nConnections = connections.size();
		w.reserve(nConnections);
		input.reserve(nConnections);
		for (int i = 0; i < c.size(); i++) {
			w[i] = rand() / RAND_MAX;
		}
	}
	DNeuron(std::vector<DNeuron> connections, thrust::device_vector<float> weights) {
		if (connections.size() < weights.size()) {
			nConnections = connections.size();
			c = connections;
			for (int i = 0; i < nConnections; i++) {
				w.push_back(weights[i]);
			}
		}
		else {
			nConnections = weights.size();
			w = weights;
			for (int i = 0; i < nConnections; i++) {
				connections.push_back(connections[i]);
			}
		}
		input.reserve(nConnections);
	}
};


void NeuronFetchInputs(SNeuron n, FetchType t);
void NeuronPushInput(SNeuron n, std::vector<float> input);
__device__ float NeuronGetValue(SNeuron n, thrust::device_vector<float> input, FetchType t); float NeuronGetValue(SNeuron n, FetchType t);
void NeuronUpdateValue(SNeuron n);
void NeuronMutate(SNeuron n);
