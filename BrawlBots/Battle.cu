#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "Bot.h"
#include "Individual.h"
#include "NeuralNetwork.h"
#include "Generation.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "device_launch_parameters.h"
#include <device_functions.h>
#include "Battle.h"
#include "Arena.h"
#include <math.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#define SMCount 13
#define NetworksPerBlock 10

__global__ void OneOnOneBattles(float* pw, float* pt, DNeuralNetwork* nnmodel, int* weightCount, int* networkCount, int* battleCount, float* resultsPersonal, float* resultsLegacy, int* nnIDs) {
	__shared__ DNeuralNetwork nnModel = *(nnmodel);
	__syncthreads();
	__shared__ DNeuralNetwork* nns;
	int plus = 0;
	if (blockIdx.x >= (*(networkCount) % SMCount)) plus = 1;
	if (threadIdx.x + plus <= (*(networkCount) / SMCount + 1) || (threadIdx.x < NetworksPerBlock && threadIdx.x < *networkCount)) {
		nns[threadIdx.x] = NNextractFromWeightArray(pw, pt, (blockIdx.x + SMCount * threadIdx.x) % *networkCount, nnModel, getID((blockIdx.x + SMCount * threadIdx.x) % *networkCount, nnIDs));
	}
	int nnCount = (int)(*(networkCount) / (SMCount + 1)) + 1 - plus;
	printf("test");
	__syncthreads();
	__shared__ int responsibleNetworks = nnCount;
	__syncthreads();
	if (responsibleNetworks == 0) return;
	
	__syncthreads();

	DNeuralNetwork contender;
	DNeuralNetwork opponent;
	int hasFight = 0;
	if (threadIdx.x < responsibleNetworks * *battleCount) {
		hasFight = 1;
		contender = nns[threadIdx.x / *battleCount];
		opponent = nns[threadIdx.x / *battleCount + threadIdx.x % (nnCount - 1)];


		float* pf1 = getPersonalFitness(resultsPersonal, contender.id, *networkCount, nnIDs);
		float* lf1 = getPersonalFitness(resultsLegacy, contender.id, *networkCount, nnIDs);
		float* pf2 = getPersonalFitness(resultsPersonal, opponent.id, *networkCount, nnIDs);
		float* lf2 = getPersonalFitness(resultsLegacy, opponent.id, *networkCount, nnIDs);
		
		int* result = battle(contender, opponent);
		if (result[1] == -2) {
			double dFitness = abs(*pf1 - *pf2);

			if (result[1] == 1) {
				//System.out.print("   CONTESTANT won.");
				double skillBoost = 0;
				if (*pf1 > *pf2) {
					skillBoost = 1 / (1 + dFitness / 3);
				}
				else {
					skillBoost = dFitness / 5;
				}
				*pf1 += (skillBoost + 1) * (result[0] / START_HEALTH * 5);
				*pf2 -= (skillBoost + 1) * (result[0] / START_HEALTH * 5);
			}
			else { //(botb.getID() == winner)
				if (result[1] == -2) {
					//System.out.print("   NOONE    won.");
				}
				else {
					//System.out.print("   " + g.ids[opponent].name + " won.");	
				}
				double skillBoost = 0;
				if (*pf2 > *pf1) {
					skillBoost = 1 / (1 + dFitness / 3);
				}
				else {
					skillBoost = dFitness / 5;
				}
				*pf2 += ((skillBoost + 1) * (result[0] / START_HEALTH * 5));
				*pf1 += ((skillBoost + 1) * (result[0] / START_HEALTH * 5 - 1));
			}

			//Update legacy fitness
			dFitness = dFitness = abs(*lf1 - *lf2);
			if (result[1] == 1) {
				double skillBoost = 0;
				if (*lf1 > *lf2) {
					skillBoost = 1 / (1 + dFitness / 3);
				}
				else {
					skillBoost = dFitness / 5;
				}
				*lf1 += (skillBoost + 1) * (result[0] / START_HEALTH * 5);
				*lf2 -= (skillBoost + 1) * (result[0] / START_HEALTH * 5);
			}
			else { //(botb.getID() == winner)
				double skillBoost = 0;
				if (*lf2 > *lf1) {
					skillBoost = 1 / (1 + dFitness / 3);
				}
				else {
					skillBoost = dFitness / 5;
				}
				*lf2 += ((skillBoost + 1) * (result[0] / START_HEALTH * 5));
				*lf1 += ((skillBoost + 1) * (result[0] / START_HEALTH * 5 - 1));
			}
		}
	}
}

__device__ int* battle(DNeuralNetwork nncontender, DNeuralNetwork nnopponent) {
	DNeuralBot b1 = DNeuralBot(&nncontender);
	DNeuralBot b2 = DNeuralBot(&nnopponent);
	DArena a = DArena(b1, b2);
	int result[2];

	int timeout = 100000;
	int count = 0;
	while (count < timeout) {
		AUpdate(a);
		if (a.winner == -2) {
			result[0] = abs(b1.health - b2.health);
			result[1] = -2;
			return  result;
		}
		if (a.winner == b1.id) {
			result[0] = abs(b1.health - b2.health);
			result[1] = 1;
			return  result;
		}
		if (a.winner == b2.id) {
			result[0] = abs(b1.health - b2.health);
			result[1] = 2;
			return  result;
		}

		count++;
	}
	result[0] = -1;
	result[1] = -1;
	return result;
}

__device__ float* getPersonalFitness(float* pf, int ID, int count, int* ids) {
	for (int i = 0; i < count; i++) {
		if (*(ids + i) == ID) return pf + i;
	}
	//Error
	return pf;
}
__device__ float* getLegacyFitness(float* lf, int ID, int count, int* ids) {
	for (int i = 0; i < count; i++) {
		if (*(ids + i) == ID) return lf + i;
	}
	//Error
	return lf;
}

void SetUpOneOnOneBattle(HGeneration g, int battlesPerIndividual) {

	thrust::host_vector<float> hpersonalFitness;
	thrust::host_vector<float> hlegacyFitness;
	thrust::host_vector<float> hnnIDs;
	thrust::host_vector<float> hweights;
	thrust::host_vector<float> hthresh;

	for (int i = 0; i < g.ids->size; i++) {
		hpersonalFitness.push_back((*g.ids)[i].personalFitness);
		hlegacyFitness.push_back((*g.ids)[i].legacyFitness);
		hnnIDs.push_back((*g.ids)[i].nnwrk.id);

		for (int j = 0; j < (*g.ids)[i].nnwrk.layers->size; i++) {
			for (int k = 0; k < (*(*g.ids)[i].nnwrk.layers)[j].neurons->size; k++) {
				hthresh.push_back((*(*(*g.ids)[i].nnwrk.layers)[j].neurons)[k].thresh);
				for (int l = 0; l < (*(*(*g.ids)[i].nnwrk.layers)[j].neurons)[k].w->size; l++) {
					hweights.push_back((*(*(*(*g.ids)[i].nnwrk.layers)[j].neurons)[k].w)[l]);
				}
			}
		}
	}


	float* h_w = thrust::raw_pointer_cast(&hweights[0]);
	float* h_t = thrust::raw_pointer_cast(&hthresh[0]);
	float* h_rp = thrust::raw_pointer_cast(&hpersonalFitness[0]);
	float* h_rl = thrust::raw_pointer_cast(&hlegacyFitness[0]);
	float* h_nnids = thrust::raw_pointer_cast(&hnnIDs[0]);;
	int weightCount = hweights.size();
	
	float* d_w;
	float* d_t;
	float* d_rp;
	float* d_rl;
	float* d_nnids;
	int * d_weightCount;
	int * d_networkCount;
	int * d_battleCount;
	DNeuralNetwork *d_nnmodel;


	for (int i = 0; i < hpersonalFitness.size; i++) {
		std::cout << i << " Personal fitness before: " << hpersonalFitness[i] << std::endl;
	}

	cudaMalloc(&d_w, sizeof(float) * hweights.size());
	cudaMalloc(&d_t, sizeof(float) * hthresh.size());
	cudaMalloc(&d_rp, sizeof(float) * hpersonalFitness.size());
	cudaMalloc(&d_rl, sizeof(float) * hlegacyFitness.size());
	cudaMalloc(&d_nnids, sizeof(int) * hnnIDs.size());
	cudaMalloc(&d_nnmodel, sizeof(HNeuralNetwork));
	cudaMalloc(&d_weightCount, sizeof(int));
	cudaMalloc(&d_networkCount, sizeof(int));
	cudaMalloc(&d_battleCount, sizeof(int));

	cudaMemcpy(d_w, h_w, sizeof(float*), cudaMemcpyHostToDevice);
	cudaMemcpy(d_t, h_t, sizeof(float*), cudaMemcpyHostToDevice);
	cudaMemcpy(d_nnmodel, &((*g.ids)[0].nnwrk), sizeof(HNeuralNetwork), cudaMemcpyHostToDevice);
	cudaMemcpy(d_weightCount, &weightCount, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_networkCount, &g.ids->size, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_battleCount, &battlesPerIndividual, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_rp, h_rp, sizeof(float*), cudaMemcpyHostToDevice);
	cudaMemcpy(d_rl, h_rl, sizeof(float*), cudaMemcpyHostToDevice);
	cudaMemcpy(d_nnids, h_nnids, sizeof(int*), cudaMemcpyHostToDevice);

	OneOnOneBattles << <13, 128 >> (d_pw, d_pt, d_nnmodel, d_wegithCount, d_networkCount, d_battleCount, d_rp, d_rl, d_nnids);

	cudaMemcpy(h_rp, d_rp, sizeof(float) * hpersonalFitness.size(), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_rl, d_rl, sizeof(float) * hlegacyFitness.size(), cudaMemcpyDeviceToHost);

	cudaFree(d_w);
	cudaFree(d_t);
	cudaFree(d_nnmodel);
	cudaFree(d_weightCount);
	cudaFree(d_networkCount);
	cudaFree(d_battleCount);
	cudaFree(d_rp);
	cudaFree(d_rl);
	cudaFree(d_nnids);

	for (int i = 0; i < hpersonalFitness.size; i++) {
		std::cout << i << " Personal fitness after: " << hpersonalFitness[i] << std::endl;
	}	
}