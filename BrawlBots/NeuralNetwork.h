#pragma once
#include "NeuralLayer.h"
#include <vector>
#include <cuda_runtime.h>
#include <thrust\device_vector.h>

struct SNeuralNetwork {
	int nIn;
	int nOut;
	int nHL; //number of hidden layers
	int sHL; //Size of each hidden layer
	int id;
	std::vector<SNeuralLayer> layers;
	std::vector<float> output;
	
	SNeuralNetwork() {
		id = rand() / RAND_MAX;
	}
	SNeuralNetwork(int nInputs, int nOutputs) {
		id = rand() / RAND_MAX;
		nIn = nInputs;
		nOut = nOutputs;
		nHL = 1;
		sHL = nIn;
		output.reserve(nOutputs);

		layers.push_back(SNeuralLayer(nIn));
		for (int i = 0; i < nHL; i++) {
			layers.push_back(SNeuralLayer(sHL, layers[i]));
		}
		layers.push_back(SNeuralLayer(nOut, nHL));
	};
};

void NNUpdate(SNeuralNetwork nn, std::vector<float> input);
__device__ void NNUpdate(SNeuralNetwork nn, thrust::device_vector<float> input);
SNeuralNetwork NNGetMutatedCopy(SNeuralNetwork nn);
SNeuralNetwork NNMutate(SNeuralNetwork nn);
SNeuralNetwork NNCopy(SNeuralNetwork nn);
__device__ SNeuralNetwork NNextractFromWeightArray(float* pw, float* pt, int nnIndex, SNeuralNetwork nn, int id);
__device__ thrust::device_vector<float> NNGetOutput(SNeuralNetwork nn, thrust::device_vector<float> input);
__device__ int getID(int index, int *ids);