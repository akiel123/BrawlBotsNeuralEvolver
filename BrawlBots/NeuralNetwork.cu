#include "NeuralNetwork.h"
#include "NeuralLayer.h"

void NNUpdate(SNeuralNetwork nn, std::vector<float> input) {
	NLPushInput(nn.layers[0], input);
	for (int i = 0; i < nn.nHL; i++) {
		NLFetchInput(nn.layers[1 + i], FetchType::Binary);
	}
	nn.output = NLGetValue(nn.layers[nn.layers.size - 1], FetchType::Sigmoid);
}
__device__ void NNUpdate(SNeuralNetwork nn, thrust::device_vector<float> input) {
	NLPushInput(nn.layers[0], input);
	for (int i = 0; i < nn.nHL; i++) {
		NLFetchInput(nn.layers[1 + i], FetchType::Binary);
	}
	nn.output = NLGetValue(nn.layers[nn.layers.size - 1], FetchType::Sigmoid);
}

__device__ SNeuralNetwork NNextractFromWeightArray(float* pw, float* pt, int nnIndex, SNeuralNetwork nn, int id) {
	float* input = pw + (nn.nIn + (nn.nIn * nn.sHL) + (nn.sHL * (nn.nHL - 1)) + (nn.sHL * nn.nOut)) * nnIndex;
	float* hl = input + nn.nIn;
	float* output = hl + (nn.nIn * nn.sHL) + (nn.sHL * (nn.nHL - 1));
	float* thresh = pt + (nn.nIn + nn.nHL * nn.sHL + nn.nOut) * nnIndex;

	SNeuralNetwork nnn = SNeuralNetwork(nn.nIn, nn.nOut);
	//Set up connected layers
	SNeuralLayer nnli = SNeuralLayer(nn.nIn);
	for (int i = 0; i < nn.nHL; i++) {
		nnn.layers[1 + i] = SNeuralLayer(nn.sHL, nnn.layers[i]);
	}
	nnn.layers[nnn.layers.size - 1] = SNeuralLayer(nn.nOut, nnn.layers.size - 2);

	//Set weights for each layer
	//Input
	for (int i = 0; i < nnn.nIn; i++) {
		nnn.layers[0].neurons[i].w.push_back(*(input) + i);
		nnn.layers[0].neurons[i].thresh = *(thresh + i);
	}

	//Hidden layer connected to input
	for (int i = 0; i < nnn.sHL; i++) {
		for (int j = 0; j < nnn.nIn; j++) {
			nnn.layers[1].neurons[i].w[j] = *(hl + i * nnn.nIn + j);
		}
		nnn.layers[1].neurons[i].thresh = *(thresh + nnn.nIn + i);
	}
	//other hidden layers
	for (int i = 1; i < nnn.nHL; i++) {
		for (int j = 0; j < nnn.sHL; j++) {
			for (int k = 0; k < nnn.sHL; k++) {
				nnn.layers[1 + i].neurons[j].w[k] = *(hl + nnn.sHL * nnn.nIn + (i - 1) * nnn.sHL * nnn.sHL + j * nnn.sHL + k);
			}
			nnn.layers[1 + i].neurons[j].thresh = *(thresh + nnn.nIn + nnn.sHL * i + j);
		}
	}
	//Output layer
	for (int i = 0; i < nnn.nOut; i++) {
		for (int j = 0; j < nnn.sHL; j++) {
			nnn.layers[nnn.layers.size - 1].neurons[i].w[j] = *(output + i * nnn.sHL + j);
		}
		nnn.layers[nnn.layers.size - 1].neurons[i].thresh = *(thresh + nnn.nIn + nnn.sHL * nnn.nHL + i);
	}
	
	nnn.id = id;

	return nnn;
}

__device__ int getID(int index, int *ids) {
	return *(ids + index);
}

__device__ thrust::device_vector<float> NNGetOutput(SNeuralNetwork nn, thrust::device_vector<float> input) {
	input = NLGetOutput(nn.layers[0], input, FetchType::Linear);
	for (int i = 0; i < nn.nHL; i++) {
		input = NLGetOutput(nn.layers[i + 1], input, FetchType::Binary);
	}
	input = NLGetOutput(nn.layers[nn.layers.size - 1], input, FetchType::Sigmoid);
	return input;
}

SNeuralNetwork NNGetMutatedCopy(SNeuralNetwork nn) {
	SNeuralNetwork child = NNCopy(nn);
	
}

SNeuralNetwork NNCopy(SNeuralNetwork nn)  {
	SNeuralNetwork nnn = SNeuralNetwork(nn.nIn, nn.nOut); //new neural network :D
	nnn.layers[0] = NLCopy(nn.layers[0]);
	for (int i = 0; i < nn.nHL; i++) {
		nnn.layers[1 + i] = NLCopy(nn.layers[1 + i], nn.layers[i]);
	}
	nnn.layers[nnn.layers.size - 1] = NLCopy(nn.layers[nn.layers.size - 1], nn.layers[nn.layers.size - 2]);
	return nnn;
}

SNeuralNetwork NNMutate(SNeuralNetwork nn) {
	for (int i = 0; i < nn.layers.size; i++) {
		NLMutate(nn.layers[i]);
	}
}