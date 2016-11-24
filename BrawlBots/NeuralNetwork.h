#pragma once
#include "NeuralLayer.h"
#include <vector>
#include <cuda_runtime.h>
#include <thrust\device_vector.h>

struct HNeuralNetwork {
	int nIn;
	int nOut;
	int nHL; //number of hidden layers
	int sHL; //Size of each hidden layer
	int id;
	std::vector<HNeuralLayer> *layers;
	std::vector<float> *output;
	
	HNeuralNetwork() {
		id = rand() / RAND_MAX;
	}
	HNeuralNetwork(int nInputs, int nOutputs) {
		id = rand() / RAND_MAX;
		nIn = nInputs;
		nOut = nOutputs;
		nHL = 1;
		sHL = nIn;
		output->reserve(nOutputs);

		layers->push_back(HNeuralLayer(nIn));
		for (int i = 0; i < nHL; i++) {
			layers->push_back(HNeuralLayer(sHL, (*layers)[i]));
		}
		layers->push_back(HNeuralLayer(nOut, nHL));
	};
};
struct DNeuralNetwork {
	int nIn;
	int nOut;
	int nHL; //number of hidden layers
	int sHL; //Size of each hidden layer
	int id;
	DNeuralLayer *layers;
	float *output;

	DNeuralNetwork() {
		id = rand() / RAND_MAX;
	}
	DNeuralNetwork(int nInputs, int nOutputs) {
		id = rand() / RAND_MAX;
		nIn = nInputs;
		nOut = nOutputs;
		nHL = 1;
		sHL = nIn;
		
		layers[0] = DNeuralLayer(nIn);
		for (int i = 0; i < nHL; i++) {
			layers[1 + i] = DNeuralLayer(sHL, layers[i]);
		}
		layers[1 + nHL] = DNeuralLayer(nOut, nHL);
	};
};

void NNUpdate(HNeuralNetwork nn, std::vector<float> input);
__device__ void NNUpdate(DNeuralNetwork nn, float* input);
HNeuralNetwork NNGetMutatedCopy(HNeuralNetwork nn);
HNeuralNetwork NNMutate(HNeuralNetwork nn);
HNeuralNetwork NNCopy(HNeuralNetwork nn);
__device__ DNeuralNetwork NNextractFromWeightArray(float* pw, float* pt, int nnIndex, DNeuralNetwork nn, int id);
__device__ float* NNGetOutput(DNeuralNetwork nn, float* input);
__device__ int getID(int index, int *ids);