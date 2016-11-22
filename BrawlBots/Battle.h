#pragma once
#include "bot.h"
#define MAX_STEPS 20000;

//Not used
struct SBattle {
	SNeuralBot b1 = SNeuralBot(&SNeuralNetwork());
	SNeuralBot b2 = SNeuralBot(&SNeuralNetwork());
	int steps = MAX_STEPS;
	int step = 0;

	SBattle(SNeuralBot bot1, SNeuralBot bot2) {
		b1 = bot1;
		b2 = bot2;
	}
};

__global__ void OneOnOneBattles(float* pw, float* pt,
	SNeuralNetwork* nnmodel, int* weightCount, int* networkCount,
	int* battleCount, float* resultsPersonal, float* resultsLegacy,
	int* nnIDs);
__device__ int* battle(SNeuralNetwork nncontender, SNeuralNetwork nnopponent);
__device__ float* getPersonalFitness(float* pf, int ID, int count, int* ids);
__device__ float* getLegacyFitness(float* lf, int ID, int count, int* ids);
